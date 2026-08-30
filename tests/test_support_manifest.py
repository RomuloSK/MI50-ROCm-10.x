import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.audit_gfx906 import audit
from scripts.validate_artifacts import validate
from scripts.validate_rocr_build import validate as validate_rocr
from scripts.verify_configure_gfx906 import verify
from scripts.mi50_policy import feature_contract, require_component, validate_environment
from scripts.write_build_provenance import write_provenance
from scripts.verify_source_lock import verify as verify_source_lock, verify_repository
from scripts.verify_patch_lock import verify as verify_patch_lock
from scripts.mi50_hardware_gate import run_gate
from scripts.mi50_runtime_validation import validate_runtime
from scripts.rocminfo_parser import parse_rocminfo
from scripts.mi50_llm_benchmark import compare_baseline, parse_throughput, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class SupportManifestTests(unittest.TestCase):
    def test_manifest_is_well_formed_and_has_required_components(self):
        manifest = json.loads((ROOT / "support-matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["target"]["llvm_target"], "gfx906")
        names = {component["name"] for component in manifest["components"]}
        for required in {"rocBLAS-Tensile", "MIOpen", "PyTorch", "llama.cpp-HIP", "opencl-runtime"}:
            self.assertIn(required, names)

    def test_feature_contract_has_hardware_truths_and_no_silent_emulation(self):
        contract = feature_contract()
        self.assertEqual(contract["llvm_target"], "gfx906")
        self.assertEqual(contract["hardware_features"]["wavefront_size"], 64)
        self.assertFalse(contract["hardware_features"]["matrix_cores"])
        self.assertFalse(contract["hardware_features"]["native_bf16"])
        self.assertFalse(contract["hardware_features"]["native_fp8"])
        self.assertEqual(contract["precision"]["bf16"], "unsupported-native")
        self.assertEqual(contract["precision"]["fp8"], "unsupported-native")
        self.assertEqual(contract["optimization_profile"]["compiler"]["offload_arch"], "gfx906")
        self.assertEqual(contract["optimization_profile"]["dispatch"]["attention"], "eager math SDPA")

    def test_feature_policy_rejects_isa_masquerading_and_mixed_targets(self):
        self.assertTrue(validate_environment({"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}))
        self.assertTrue(validate_environment({"THEROCK_AMDGPU_FAMILIES": "gfx906;gfx1100"}))
        self.assertTrue(validate_environment({"PYTORCH_ROCM_ARCH": "gfx906,gfx1100"}))
        self.assertTrue(validate_environment({"CMAKE_HIP_ARCHITECTURES": "gfx906;gfx1030"}))
        self.assertEqual(validate_environment({"THEROCK_AMDGPU_FAMILIES": "gfx906"}), [])
        self.assertEqual(validate_environment({"PYTORCH_ROCM_ARCH": "gfx906"}), [])

    def test_provenance_records_locked_sources_and_patch_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "build-provenance.json"
            payload = write_provenance(output, repository_root=ROOT)
            self.assertEqual(payload["project_version"], "10.0.0+mi50.5")
            self.assertEqual(payload["target"], "gfx906")
            self.assertIn("container", payload)
            self.assertIn("sha256:", payload["container"]["base_image"])
            self.assertTrue(payload["patches_observed"])
            self.assertIn("MI50_ENABLE_OPENCL", payload["environment"])
            self.assertIn("MI50_BUILD_TESTING", payload["environment"])
            self.assertIn("MI50_BUILD_PROFILE", payload["environment"])
            self.assertIn("MI50_BUILD_PYTHON_PACKAGES", payload["environment"])
            self.assertIn("ROCCLR_ENABLE_OPENGL", payload["environment"])
            self.assertTrue(output.is_file())

    def test_hardware_gate_is_pending_without_kfd_and_never_accepts_override(self):
        report = run_gate()
        self.assertIn(report["status"], {"GPU-test-pending", "fail"})
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}):
            report = run_gate()
        self.assertEqual(report["status"], "fail")
        self.assertIn("ISA override", report["errors"][0])

    def test_runtime_validation_is_pending_without_kfd(self):
        report = validate_runtime()
        self.assertIn(report["status"], {"GPU-test-pending", "fail"})
        with patch.dict(os.environ, {"ROCR_OVERRIDE_GFX_VERSION": "10.3.0"}):
            report = validate_runtime()
        self.assertEqual(report["status"], "fail")
        self.assertIn("ISA override", report["errors"][0])

    def test_rocminfo_parser_enforces_native_gfx906_wave64_contract(self):
        report = parse_rocminfo(
            """
            Name:                    gfx906
            Wavefront Size:          64
            Name:                    gfx942
            Wavefront Size:          32
            Name:                    amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-
            """
        )
        self.assertTrue(report["has_native_gfx906"])
        self.assertTrue(report["wavefront64_observed"])
        self.assertEqual(report["native_gfx906_agent_count"], 1)
        self.assertIn(32, report["wavefront_sizes"])

    def test_source_lock_is_pinned(self):
        lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["rocm_version"], "10.0.0")
        self.assertFalse(lock["policy"]["allow_gfx_override"])
        self.assertEqual(lock["upstream_streams"]["stable_release_base"], "10.0")
        self.assertEqual(lock["upstream_streams"]["newer_candidate_stream"], "10.1-nightly")
        self.assertIn("commit-pinned", lock["upstream_streams"]["candidate_policy"])
        self.assertGreaterEqual(len(lock["repositories"]), 4)
        self.assertEqual(lock["build_dependencies"]["python"]["CppHeaderParser"], "2.7.4")
        self.assertEqual(lock["build_dependencies"]["python"]["msgpack"], "1.1.1")
        self.assertEqual(lock["build_dependencies"]["python"]["zstandard"], "0.25.0")
        self.assertEqual(lock["build_dependencies"]["python"]["pytest"], "8.4.2")
        self.assertEqual(lock["build_dependencies"]["python"]["pytest-subtests"], "0.15.0")
        self.assertIn("libx11-dev", lock["build_dependencies"]["system_packages"])
        for repository in lock["repositories"]:
            self.assertRegex(repository["commit"], r"^[0-9a-f]{40}$")
        therock = next(repository for repository in lock["repositories"] if repository["name"] == "TheRock")
        self.assertEqual(therock["commit"], "16adc4d875fd4f65ea23c7c84e1c66706fde3047")
        self.assertEqual(therock["tag_object"], "fcd02232ce4a2ff4f96e6155c0a63c7b5b8d438f")
        for patch in lock.get("patches", []):
            patch_path = ROOT / patch["file"]
            self.assertTrue(patch_path.is_file())
            digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
            self.assertEqual(digest, patch["sha256"])
        self.assertTrue(any("hipblaslt-register-gfx906" in patch["file"] for patch in lock["patches"]))
        self.assertTrue(any("rocblas-no-abort-missing-tensile" in patch["file"] for patch in lock["patches"]))
        self.assertTrue(any("rocclr-optional-opengl" in patch["file"] for patch in lock["patches"]))
        self.assertTrue(any("rocblas-no-abort-get-solutions" in patch["file"] for patch in lock["patches"]))
        self.assertTrue(any("rocblas-no-abort-no-device" in patch["file"] for patch in lock["patches"]))

    def test_patch_queue_is_parseable(self):
        for patch in sorted((ROOT / "patches").glob("*.patch")):
            result = subprocess.run(
                ["git", "apply", "--stat", str(patch)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"invalid patch {patch}: {result.stderr}")

    def test_forward_port_mode_is_wired_into_build(self):
        patch = (ROOT / "patches/0001-gfx906-forward-port-target-policy.patch").read_text(
            encoding="utf-8"
        )
        build_script = (ROOT / "scripts/build_therock_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("MI50_ENABLE_FORWARD_PORTS", patch)
        self.assertIn("list(REMOVE_ITEM _gfx906_excluded_projects", patch)
        for candidate in ("hipBLASLt", "hipSPARSELt", "composable_kernel", "hipTensor"):
            self.assertIn(candidate, patch)
        hipblaslt_patch = (ROOT / "patches/0006-hipblaslt-register-gfx906-target.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('"gfx906"', hipblaslt_patch)
        self.assertIn("tensilelite_supported_architectures.cmake", hipblaslt_patch)
        hipsparselt_patch = (ROOT / "patches/0007-hipsparselt-gfx906-fallback-policy.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("hipSPARSELt", hipsparselt_patch)
        self.assertIn("MFMA", hipsparselt_patch)
        rocblas_patch = (ROOT / "patches/0008-rocblas-no-abort-missing-tensile.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("projects/rocblas/library/src/tensile_host.cpp", rocblas_patch)
        self.assertIn("rocblas_status_not_implemented", rocblas_patch)
        self.assertIn("return m_libraryMap[processor]", rocblas_patch)
        rocblas_dispatch_patch = (ROOT / "patches/0009-rocblas-gfx906-hipblaslt-fallback.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROCBLAS_USE_HIPBLASLT=1", rocblas_dispatch_patch)
        self.assertIn("rocblas_internal_get_arch(prob.handle) == 906", rocblas_dispatch_patch)
        rocclr_patch = (ROOT / "patches/0010-rocclr-optional-opengl-for-hip.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROCCLR_ENABLE_OPENGL", rocclr_patch)
        solutions_patch = (ROOT / "patches/0012-rocblas-no-abort-get-solutions.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("!library || !deviceProp", solutions_patch)
        self.assertIn("-DMI50_ENABLE_FORWARD_PORTS=ON", build_script)
        self.assertIn("-DMI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=OFF", build_script)
        # OpenCL/ocl-clr is intentionally outside the Linux-first inference
        # deliverable; keep the optional host OpenGL dependency from entering
        # a reproducible MI50 build through cached defaults.
        self.assertIn("MI50_ENABLE_OPENCL", build_script)
        self.assertIn("-DTHEROCK_ENABLE_OCL_RUNTIME=${ENABLE_OPENCL}", build_script)
        self.assertIn("-DTHEROCK_ENABLE_OCL_ICD=${ENABLE_OPENCL}", build_script)
        self.assertIn("-DTHEROCK_ENABLE_CORE_RUNTIME_TESTS=OFF", build_script)
        self.assertIn("ROCCLR_ENABLE_OPENGL", build_script)
        self.assertIn("MI50_BUILD_TESTING", build_script)
        self.assertIn("MI50_BUILD_PROFILE", build_script)
        self.assertIn("THEROCK_ENABLE_STORAGE_LIBS=OFF", build_script)
        self.assertIn("THEROCK_ENABLE_PROFILER=OFF", build_script)
        self.assertIn("--skip-audit", build_script)
        self.assertNotIn("-DROCR_TARGET_DEVICES=gfx906", build_script)
        self.assertIn("apply_patch_queue()", build_script)
        self.assertIn("--ignore-space-change --reverse --check", build_script)
        self.assertIn("superseded", build_script)
        self.assertIn("0010-rocclr-optional-opengl-for-hip.patch", build_script)
        self.assertIn("0011-rocclr-opengl-env-override.patch", build_script)
        self.assertIn('git ls-remote --exit-code --heads "$url" "refs/heads/${ref}"', build_script)
        self.assertIn("fetch --no-tags origin \"$commit\"", build_script)
        self.assertIn("validate_rocr_build.py", build_script)
        self.assertIn("mi50_features.py\" --check-environment", build_script)
        self.assertIn("source-lock-verification.json\" \"${ARTIFACT_DIR}", build_script)
        self.assertIn("patch-lock-verification.json\" \"${ARTIFACT_DIR}", build_script)
        self.assertIn("a real\n  # packaging failure", build_script)
        self.assertNotIn("--target therock-dist --parallel \"$JOBS\" || true", build_script)

        pytorch_script = (ROOT / "scripts/build/pytorch/build_pytorch_gfx906.sh").read_text(
            encoding="utf-8"
        )
        llama_script = (ROOT / "scripts/build/llama.cpp/build_llama_gfx906.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"', pytorch_script)
        self.assertIn('ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"', llama_script)

        rocr_script = (ROOT / "scripts/build_rocr_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("-DROCR_TARGET_DEVICES=gfx906", rocr_script)
        self.assertIn("validate_rocr_build.py", rocr_script)
        self.assertIn("ROCR_TARGET_DEVICES_GFX906", rocr_script)
        self.assertIn("rocr_host_smoke.py", rocr_script)
        self.assertIn("mi50_features.py\" --check-environment", rocr_script)

        superbuild_patch = (ROOT / "patches/0005-therock-forward-rocr-target-args.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROCR_TARGET_DEVICES=gfx906", superbuild_patch)
        self.assertIn("THEROCK_DIST_AMDGPU_FAMILIES", superbuild_patch)

        matrix = json.loads((ROOT / "support-matrix.json").read_text(encoding="utf-8"))
        by_name = {component["name"]: component for component in matrix["components"]}
        for candidate in ("hipBLASLt", "composable-kernel", "hipTensor"):
            self.assertEqual(by_name[candidate]["status"], "forward-port-candidate")
        self.assertEqual(by_name["hipSPARSELt"]["status"], "unsupported-on-gfx906")
        with self.assertRaises(RuntimeError):
            require_component("hipSPARSELt")

    def test_native_hip_compile_smoke_is_targeted_and_host_only(self):
        source = (ROOT / "tests/hip/gfx906_compile_smoke.hip").read_text(encoding="utf-8")
        script = (ROOT / "scripts/hip_compile_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("does not call hipInit", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", script)

    def test_native_hip_runtime_smoke_is_targeted_and_pending_without_hardware(self):
        script = (ROOT / "scripts/run_hip_runtime_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_runtime_smoke.hip").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("/dev/kfd", script)
        self.assertIn("hipGetDeviceCount", source)
        self.assertIn("hipMalloc", source)
        self.assertIn("hipMemcpyAsync", source)
        self.assertIn("hipStreamSynchronize", source)
        self.assertIn("hipEventElapsedTime", source)
        self.assertIn("gfx906", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        self.assertIn("run_mi50_device_matrix_smoke.sh", host_runner)

    def test_builder_checks_artifact_splitter_python_dependencies(self):
        builder = (ROOT / "scripts/build_therock_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("import msgpack, zstandard", builder)
        self.assertIn("msgpack==1.1.1", builder)
        self.assertIn("zstandard==0.25.0", builder)
        self.assertIn("command -v tclsh", builder)
        self.assertIn("THEROCK_ENABLE_HIPTENSOR=OFF", builder)
        self.assertIn("THEROCK_ENABLE_COMPOSABLE_KERNEL=OFF", builder)
        self.assertIn("THEROCK_ENABLE_ROCWMMA=OFF", builder)
        self.assertIn("THEROCK_ENABLE_HIPSPARSELT=OFF", builder)
        self.assertIn("THEROCK_ENABLE_ROCPROFILER_COMPUTE=OFF", builder)
        self.assertIn("ROCROLLER_BUILD_TESTING=OFF", builder)
        self.assertIn("MI50_BUILD_PYTHON_PACKAGES", builder)
        self.assertIn("validate_python_packages.py", builder)
        self.assertIn("filter_rocm_wheels.py", builder)

    def test_device_matrix_smoke_is_native_and_peer_aware(self):
        script = (ROOT / "scripts/run_mi50_device_matrix_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_device_matrix_smoke.hip").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("hipDeviceCanAccessPeer", source)
        self.assertIn("hipDeviceEnablePeerAccess", source)
        self.assertIn("warpSize", source)
        self.assertIn("hipMemcpy", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_builder_declares_nested_hip_dependencies(self):
        dockerfile = (ROOT / "containers/ubuntu-24.04/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("CppHeaderParser==2.7.4", dockerfile)
        self.assertIn("joblib==1.5.1", dockerfile)
        self.assertIn("msgpack==1.1.1", dockerfile)
        self.assertIn("zstandard==0.25.0", dockerfile)
        self.assertIn("pytest==8.4.2", dockerfile)
        self.assertIn("pytest-subtests==0.15.0", dockerfile)
        self.assertIn("BASE_IMAGE=ubuntu:24.04@sha256:", dockerfile)
        for package in (
            "libegl-dev",
            "libgl-dev",
            "libglx-dev",
            "libx11-dev",
            "libsqlite3-dev",
            "gfortran",
            "python3-magic",
            "rustup",
            "tcl",
            "texinfo",
        ):
            self.assertIn(package, dockerfile)
        self.assertIn('ARG RUST_TOOLCHAIN=1.98.0', dockerfile)

    def test_container_builder_records_resolved_image_identity(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("out", dockerignore)
        self.assertIn("sources", dockerignore)
        self.assertIn("build", dockerignore)
        script = (ROOT / "scripts/build_container.sh").read_text(encoding="utf-8")
        self.assertIn("MI50_BASE_IMAGE", script)
        self.assertIn("MI50_CONTAINER_METADATA", script)
        self.assertIn("--iidfile", script)
        provenance = (ROOT / "scripts/write_container_provenance.py").read_text(encoding="utf-8")
        self.assertIn('"docker", "image", "inspect"', provenance)
        self.assertIn("repo_digests", provenance)

    def test_source_lock_verifier_rejects_unpinned_or_missing_checkouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "repositories": [
                            {"name": "TheRock", "commit": "a" * 40},
                            {"name": "rocm-systems", "commit": "b" * 40},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = verify_source_lock(root, lock)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any(item.startswith("TheRock: no git checkout") for item in report["missing"]))
            self.assertTrue(any(item.startswith("rocm-systems: no git checkout") for item in report["missing"]))
        build_script = (ROOT / "scripts/build_therock_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("verify_source_lock.py", build_script)
        self.assertIn("source-lock-verification.json", build_script)

    def test_standalone_rocr_builder_verifies_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(
                json.dumps({"repositories": [{"name": "rocm-systems", "commit": "c" * 40}]}),
                encoding="utf-8",
            )
            report = verify_repository(root, "rocm-systems", lock)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(report["missing"])
        script = (ROOT / "scripts/build_rocr_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("source-repo-root", script)
        self.assertIn("--repository-name rocm-systems", script)
        self.assertIn("verify_patch_lock.py", script)
        self.assertIn('"${INSTALL_PREFIX}/source-lock-verification.json"', script)
        self.assertIn('"${INSTALL_PREFIX}/gfx906-rocr-validation.json"', script)

    def test_patch_lock_verifier_matches_all_recorded_digests(self):
        report = verify_patch_lock(ROOT, ROOT / "sources.lock.json")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["mismatched"], [])
        self.assertGreaterEqual(len(report["patches"]), 13)
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "patches": [
                            {"file": "patches/0001-gfx906-forward-port-target-policy.patch", "sha256": "0" * 64}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bad_report = verify_patch_lock(ROOT, lock)
            self.assertEqual(bad_report["status"], "fail")
            self.assertEqual(len(bad_report["mismatched"]), 1)
        build_script = (ROOT / "scripts/build_therock_gfx906.sh").read_text(encoding="utf-8")
        self.assertIn("verify_patch_lock.py", build_script)
        self.assertIn("patch-lock-verification.json", build_script)

    def test_windows_wsl_documentation_keeps_runtime_claim_honest(self):
        document = (ROOT / "docs/WINDOWS_WSL.md").read_text(encoding="utf-8")
        self.assertIn("GPU-test-pending", document)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION", document)
        self.assertIn("Read-only file system", document)
        self.assertIn("native Windows ROCm runtime is", document)

    def test_llm_benchmark_parser_and_regression_gate(self):
        metrics = parse_throughput(
            "prompt eval time = 10.0 ms / 128 tokens (12800.0 t/s)\n"
            "eval time = 20.0 ms / 64 runs (3200.0 t/s)\n"
        )
        self.assertEqual(metrics["prompt_tokens_per_second"], 12800.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 3200.0)
        self.assertEqual(compare_baseline(metrics, {"decode_tokens_per_second": 3300.0}), [])
        regressions = compare_baseline(metrics, {"decode_tokens_per_second": 4000.0})
        self.assertEqual(regressions[0]["metric"], "decode_tokens_per_second")

    def test_llm_benchmark_is_pending_before_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"test")
            report = run_benchmark(model=model)
            self.assertEqual(report["status"], "GPU-test-pending")
            self.assertIn("benchmark_command", report)

    def test_audit_finds_target_and_override_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.cmake").write_text("set(TARGET gfx906)\n", encoding="utf-8")
            report = audit(root, set())
            self.assertTrue(report["policy"]["gfx906_seen"])
            self.assertFalse(report["policy"]["gfx_override_seen"])

            (root / "bad.sh").write_text("export HSA_OVERRIDE_GFX_VERSION=9.0.8\n", encoding="utf-8")
            report = audit(root, set())
            self.assertTrue(report["policy"]["gfx_override_seen"])
            self.assertTrue(report["policy"]["gfx_override_assignments"])

    def test_audit_tolerates_upstream_reads_of_the_override_variable(self):
        # rocm_agent_enumerator legitimately reads the variable to warn the
        # user; treating that as masquerading would fail every correct build.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tool.py").write_text(
                "gfx_override = os.environ.get(\"HSA_OVERRIDE_GFX_VERSION\")\n"
                "print('Invalid HSA_OVERRIDE_GFX_VERSION value')\n"
                "unset HSA_OVERRIDE_GFX_VERSION\n",
                encoding="utf-8",
            )
            report = audit(root, set())
            self.assertFalse(report["policy"]["gfx_override_seen"])
            self.assertEqual(report["policy"]["gfx_override_mentions"], 3)
            self.assertEqual(report["policy"]["gfx_override_assignments"], [])

    def test_isa_override_detector_covers_the_real_install_forms(self):
        from scripts.mi50_policy import isa_override_findings

        enabling = [
            "export HSA_OVERRIDE_GFX_VERSION=9.0.8",
            'cmake_args+=("-DROCR_OVERRIDE_GFX_VERSION=10.3.0")',
            'os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.8"',
            'putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.8");',
            '{"HSA_OVERRIDE_GFX_VERSION": "9.0.8"}',
        ]
        harmless = [
            "if (getenv(\"HSA_OVERRIDE_GFX_VERSION\") != NULL) warn();",
            'value = os.environ.get("HSA_OVERRIDE_GFX_VERSION")',
            "unset HSA_OVERRIDE_GFX_VERSION",
            "HSA_OVERRIDE_GFX_VERSION=  # cleared",
            "print('Invalid HSA_OVERRIDE_GFX_VERSION value')",
        ]
        for line in enabling:
            self.assertTrue(isa_override_findings(line), msg=f"missed: {line}")
        for line in harmless:
            self.assertEqual(isa_override_findings(line), [], msg=f"false positive: {line}")

    def test_artifact_validator_requires_device_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dist_info.json").write_text('{"target":"gfx906"}\n', encoding="utf-8")
            (root / "TensileLibrary_lazy_gfx906.dat").write_bytes(b"placeholder")
            (root / "gfx906.kdb").write_bytes(b"placeholder")
            report = validate(root)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["missing"], [])

            (root / "build-provenance.json").write_text(
                json.dumps({"target": "gfx906", "runtime_claim": "artifact-only; GPU execution remains pending-hardware"}),
                encoding="utf-8",
            )
            (root / "source-lock-verification.json").write_text(
                json.dumps({"status": "pass"}), encoding="utf-8"
            )
            (root / "patch-lock-verification.json").write_text(
                json.dumps({"status": "pass"}), encoding="utf-8"
            )
            report = validate(root)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["checks"]["provenance_target"])
            self.assertTrue(report["checks"]["source_lock_status"])
            self.assertTrue(report["checks"]["patch_lock_status"])

            (root / "build-provenance.json").write_text(
                json.dumps({"target": "gfx1100", "runtime_claim": "artifact-only"}),
                encoding="utf-8",
            )
            report = validate(root)
            self.assertEqual(report["status"], "fail")
            self.assertIn("provenance_target", report["missing"])
            (root / "build-provenance.json").unlink()

            (root / "mi50_features.json").write_text(
                json.dumps(
                    {
                        "llvm_target": "gfx906",
                        "hardware_features": {
                            "native_bf16": False,
                            "native_fp8": False,
                            "matrix_cores": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = validate(root)
            self.assertEqual(report["status"], "pass")

            (root / "mi50_features.json").write_text(
                '{"llvm_target":"gfx906","hardware_features":{"native_fp8":true}}',
                encoding="utf-8",
            )
            report = validate(root)
            self.assertEqual(report["status"], "fail")
            self.assertIn("feature_policy_no_native_fp8", report["missing"])

    def test_configure_verifier_accepts_forward_port_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeCache.txt").write_text(
                "MI50_ENABLE_FORWARD_PORTS:BOOL=ON\n"
                "THEROCK_AMDGPU_FAMILIES:STRING=gfx906\n"
                "THEROCK_DIST_AMDGPU_FAMILIES:STRING=gfx906\n"
                "THEROCK_TEST_AMDGPU_FAMILIES:STRING=gfx906\n",
                encoding="utf-8",
            )
            (root / "artifact_subprojects.json").write_text(
                json.dumps(
                    {
                        "blas": ["rocBLAS", "hipBLASLt", "hipSPARSELt"],
                        "hipblaslt": ["hipBLASLt"],
                        "composable-kernel": ["composable_kernel"],
                        "hiptensor": ["hipTensor"],
                        "miopen": ["MIOpen"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(verify(root)["status"], "pass")

    def test_rocr_validator_requires_target_scoped_device_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libhsa-runtime64.a").write_bytes(b"archive")
            (root / "kCodeTrapHandlerV2_9.hsaco").write_bytes(b"ELF")
            for name in (
                "kCodeCopyAligned9.hsaco",
                "kCodeCopyMisaligned9.hsaco",
                "kCodeFill9.hsaco",
            ):
                (root / name).write_bytes(b"ELF")
            image = root / "ocl_blit_object_gfx906"
            image.write_bytes(b"amdhsa.target: amdgcn-amd-amdhsa--gfx906")
            report = validate_rocr(root)
            self.assertEqual(report["status"], "pass")

            (root / "kCodeFill10.hsaco").write_bytes(b"wrong target")
            report = validate_rocr(root)
            self.assertEqual(report["status"], "fail")
            self.assertIn("no_newer_isa_objects", report["missing"])


if __name__ == "__main__":
    unittest.main()
