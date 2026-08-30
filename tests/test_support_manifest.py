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
from scripts.mi50_doctor import command_version, resolve_rocm_root
from scripts.mi50_runtime_validation import runtime_environment as runtime_validation_environment, validate_runtime
from scripts.rocminfo_parser import parse_rocminfo
from scripts.mi50_llm_benchmark import command_path, compare_baseline, parse_throughput, run_benchmark, runtime_environment
from scripts.mi50_validation_suite import STEPS, run_step, run_suite


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

    def test_hardware_gate_fails_when_a_diagnostic_command_returns_nonzero(self):
        captured_environments = []

        def fake_command(command, *, environment=None):
            captured_environments.append(environment)
            if command[0] == "hipconfig":
                return {"command": command, "status": "fail", "returncode": 1, "stdout": "", "stderr": "broken"}
            return {
                "command": command,
                "status": "pass",
                "returncode": 0,
                "stdout": "Name: gfx906\nWavefront Size: 64\n" if command[0] == "rocminfo" else "",
                "stderr": "",
            }

        ready = {"status": "ready-for-rocr", "errors": []}
        with tempfile.TemporaryDirectory() as directory:
            rocm_root = Path(directory)
            (rocm_root / "bin").mkdir()
            with patch("scripts.mi50_hardware_gate.Path.exists", return_value=True), patch(
                "scripts.mi50_hardware_gate.collect_readiness", return_value=ready
            ), patch("scripts.mi50_hardware_gate.run_command", side_effect=fake_command):
                report = run_gate(rocm_path=str(rocm_root))
        self.assertEqual(report["status"], "fail")
        self.assertIn("diagnostic command failed: hipconfig --full", report["errors"])
        self.assertTrue(captured_environments[0]["PATH"].startswith(str(rocm_root / "bin")))

    def test_runtime_validation_scopes_requested_rocm_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            rocm_root = Path(directory)
            for relative in ("bin", "lib", "lib/llvm/bin", "lib/llvm/lib", "lib/rocm_sysdeps/lib"):
                (rocm_root / relative).mkdir(parents=True, exist_ok=True)
            environment = runtime_validation_environment(str(rocm_root))
            expected_root = str(rocm_root.resolve())
        self.assertEqual(environment["ROCM_PATH"], expected_root)
        self.assertTrue(environment["PATH"].startswith(str(rocm_root / "bin")))
        self.assertTrue(environment["LD_LIBRARY_PATH"].startswith(str(rocm_root / "lib")))

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
        self.assertTrue(any("rocr-fallback-to-static-kfd" in patch["file"] for patch in lock["patches"]))
        self.assertTrue(any("rocr-fallback-on-incomplete-dxg-api" in patch["file"] for patch in lock["patches"]))

    def test_patch_queue_is_parseable(self):
        for patch in sorted((ROOT / "patches").rglob("*.patch")):
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
        rocr_loader_patch = (ROOT / "patches/0027-rocr-fallback-to-static-kfd.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("is_loaded_ = false", rocr_loader_patch)
        self.assertIn("is_wsl_dxg_ = false", rocr_loader_patch)
        self.assertIn("statically linked Linux/KFD thunk", rocr_loader_patch)
        rocr_incomplete_dxg_patch = (ROOT / "patches/0028-rocr-fallback-on-incomplete-dxg-api.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("falling back to the native Linux KFD thunk", rocr_incomplete_dxg_patch)
        self.assertIn("CloseLib(thunk_handle)", rocr_incomplete_dxg_patch)
        self.assertIn("LoadThunkApiTable();", rocr_incomplete_dxg_patch)
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
        self.assertIn("validate_elf_dependencies.py", build_script)
        self.assertIn("RDC_TEST_RPATH", build_script)
        self.assertIn("patchelf", build_script)
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
        self.assertIn("ROCM_PATH_HINT", rocr_script)
        self.assertIn("lib/llvm/amdgcn/bitcode", rocr_script)
        self.assertIn("export CC=", rocr_script)

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

    def test_experimental_hipblaslt_builder_isolated_and_rejects_empty_catalogs(self):
        script = (
            ROOT / "scripts/build/pytorch/build_hipblaslt_gfx906_experimental.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("-DGPU_TARGETS=gfx906", script)
        self.assertIn("-DHIPBLASLT_ENABLE_DEVICE=ON", script)
        self.assertIn("-DHIPBLASLT_ENABLE_EXTOPS=OFF", script)
        self.assertIn("-DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF", script)
        self.assertIn("-DHIPBLASLT_ENABLE_CLIENT=OFF", script)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION is forbidden", script)
        self.assertIn("no executable .text section", script)
        self.assertIn("exit 78", script)
        lock = json.loads((ROOT / "downstream.lock.json").read_text(encoding="utf-8"))
        experiment = lock["components"]["hipblaslt_gfx906_experimental"]
        self.assertEqual(experiment["build_policy"]["GPU_TARGETS"], "gfx906")
        self.assertEqual(experiment["result"], "rejected-empty-gfx906-kernel-catalog")

    def test_pytorch_precision_patch_is_locked_and_wired_into_build(self):
        lock = json.loads((ROOT / "downstream.lock.json").read_text(encoding="utf-8"))
        records = lock["components"]["pytorch"]["downstream_patches"]
        self.assertEqual(len(records), 2)
        for record in records:
            patch_path = ROOT / record["file"]
            self.assertTrue(patch_path.is_file())
            self.assertEqual(record["sha256"], hashlib.sha256(patch_path.read_bytes()).hexdigest())
        bf16_patch = (ROOT / records[0]["file"]).read_text(encoding="utf-8")
        self.assertIn("is_bf16_supported", bf16_patch)
        self.assertIn("arch.startswith(\"gfx906\")", bf16_patch)
        self.assertIn("return False", bf16_patch)
        tf32_patch = (ROOT / records[1]["file"]).read_text(encoding="utf-8")
        self.assertIn("is_tf32_supported", tf32_patch)
        self.assertIn("if not is_available()", tf32_patch)

        build_script = (ROOT / "scripts/build/pytorch/build_pytorch_gfx906.sh").read_text(
            encoding="utf-8"
        )
        metadata_script = (ROOT / "scripts/build/pytorch/write_pytorch_metadata.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("apply_downstream_patches", build_script)
        self.assertIn("PYTORCH_MI50_PATCH_DIR", build_script)
        self.assertIn("--patch-dir", build_script)
        self.assertIn("downstream_patches", metadata_script)
        self.assertIn("hashlib.sha256", metadata_script)

    def test_installer_is_target_scoped_and_atomic(self):
        installer = (ROOT / "scripts/install_rocm_mi50.py").read_text(encoding="utf-8")
        for marker in (
            "extractall(staging, filter=\"data\")",
            "archive is missing required gfx906 payload",
            "timestamped",
            "mi50-env.sh",
            "lib/rocm_sysdeps/lib",
            "lib/llvm/lib",
            "HSA_OVERRIDE_GFX_VERSION",
            "rocm/lib/rocblas/library/",
            "rocm/share/miopen/db/",
        ):
            self.assertIn(marker, installer)

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

    def test_hip_graph_smoke_isolated_and_native_targeted(self):
        script = (ROOT / "scripts/run_mi50_graph_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_runtime_smoke.hip").read_text(encoding="utf-8")
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("--graph", script)
        self.assertIn("hipStreamBeginCapture", source)
        self.assertIn("hipStreamEndCapture", source)
        self.assertIn("hipGraphInstantiate", source)
        self.assertIn("hipGraphLaunch", source)
        self.assertIn("run_mi50_graph_smoke.sh", host_runner)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_hiprtc_smoke_compiles_and_targets_native_gfx906(self):
        script = (ROOT / "scripts/run_mi50_hiprtc_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_hiprtc_smoke.cpp").read_text(encoding="utf-8")
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lhiprtc", script)
        self.assertIn("hiprtcCompileProgram", source)
        self.assertIn("--gpu-architecture=gfx906", source)
        self.assertIn("hipModuleLoadData", source)
        self.assertIn("hipModuleLaunchKernel", source)
        self.assertIn("GPU-test-pending", source)
        self.assertIn("run_mi50_hiprtc_smoke.sh", host_runner)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_hipblas_smoke_exercises_stable_rocblas_route(self):
        script = (ROOT / "scripts/run_mi50_hipblas_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_hipblas_smoke.cpp").read_text(encoding="utf-8")
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lhipblas", script)
        self.assertIn("ROCBLAS_USE_HIPBLASLT=0", script)
        self.assertIn("hipblasSgemm", source)
        self.assertIn("hipblasCreate", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("run_mi50_hipblas_smoke.sh", host_runner)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_validation_suite_is_ordered_and_preserves_pending_status(self):
        from unittest.mock import patch

        self.assertEqual(STEPS[0][0], "kernel-readiness")
        self.assertEqual(STEPS[-1][0], "RCCL")

        def fake_step(command, *, environment, timeout):
            return {"status": "GPU-test-pending", "returncode": 77, "stdout": "", "stderr": ""}

        with patch("scripts.mi50_validation_suite.run_step", side_effect=fake_step):
            report = run_suite(rocm_path="/opt/rocm-mi50", require_gpu=False, timeout=1)
        self.assertEqual(report["status"], "GPU-test-pending")
        self.assertEqual(report["summary"]["pending"], len(STEPS))
        self.assertEqual(len(report["steps"]), len(STEPS))

        with patch("scripts.mi50_validation_suite.run_step", side_effect=fake_step):
            required = run_suite(rocm_path="/opt/rocm-mi50", require_gpu=True, timeout=1)
        self.assertEqual(required["status"], "fail")

    def test_validation_suite_scopes_tool_and_library_paths_to_requested_rocm(self):
        from unittest.mock import patch

        captured = []

        def fake_step(command, *, environment, timeout):
            captured.append(environment)
            return {"status": "GPU-test-pending", "returncode": 77, "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as directory:
            rocm_root = Path(directory)
            for relative in ("bin", "lib", "lib/llvm/bin", "lib/llvm/lib", "lib/rocm_sysdeps/lib"):
                (rocm_root / relative).mkdir(parents=True, exist_ok=True)
            with patch("scripts.mi50_validation_suite.run_step", side_effect=fake_step):
                run_suite(rocm_path=str(rocm_root), require_gpu=False, timeout=1)

        self.assertTrue(captured)
        environment = captured[0]
        self.assertEqual(environment["ROCM_PATH"], str(rocm_root.resolve()))
        self.assertTrue(environment["PATH"].startswith(str(rocm_root / "bin")))
        self.assertTrue(environment["LD_LIBRARY_PATH"].startswith(str(rocm_root / "lib")))

    def test_host_runner_scopes_requested_rocm_paths(self):
        runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        self.assertIn('export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"', runner)
        self.assertIn("lib/rocm_sysdeps/lib", runner)
        self.assertIn("lib/llvm/lib", runner)

    def test_doctor_uses_tools_from_selected_artifact_root(self):
        from subprocess import CompletedProcess

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            rocm_root = prefix / "rocm"
            (rocm_root / "bin").mkdir(parents=True)
            tool = rocm_root / "bin" / "rocminfo"
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o755)
            resolved = resolve_rocm_root(prefix)
            self.assertEqual(resolved, rocm_root.resolve())
            with patch("scripts.mi50_doctor.subprocess.run") as run:
                run.return_value = CompletedProcess([str(tool), "--version"], 0, "artifact-tool\n", "")
                self.assertEqual(command_version("rocminfo", rocm_root=resolved), "artifact-tool")
            self.assertEqual(run.call_args.args[0][0], str(tool))
            with patch("scripts.mi50_doctor.subprocess.run") as run:
                run.return_value = CompletedProcess([str(tool), "--version"], 1, "partial\n", "broken\n")
                self.assertIn("version query failed", command_version("rocminfo", rocm_root=resolved))

    def test_downstream_build_wrappers_prefer_packaged_llvm(self):
        for relative in (
            "scripts/build/pytorch/build_pytorch_gfx906.sh",
            "scripts/build/llama.cpp/build_llama_gfx906.sh",
        ):
            script = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("${ROCM_ROOT}/lib/llvm/bin", script)
            self.assertIn("lib/rocm_sysdeps/lib", script)
            self.assertIn("lib/llvm/lib", script)
            self.assertIn("export LD_LIBRARY_PATH", script)

    def test_validation_suite_uses_json_status_when_gate_returns_zero(self):
        from subprocess import CompletedProcess
        from unittest.mock import patch

        completed = CompletedProcess(
            ["python", "gate.py"],
            0,
            stdout=json.dumps({"status": "GPU-test-pending"}),
            stderr="",
        )
        with patch("scripts.mi50_validation_suite.subprocess.run", return_value=completed):
            report = run_step(["python", "gate.py"], environment=os.environ.copy(), timeout=1)
        self.assertEqual(report["status"], "GPU-test-pending")

    def test_rccl_smoke_is_native_and_requires_two_gfx906_devices(self):
        script = (ROOT / "scripts/run_mi50_rccl_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/rccl/mi50_rccl_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("/dev/kfd", script)
        self.assertIn("-lrccl", script)
        self.assertIn("ncclCommInitRank", source)
        self.assertIn("ncclAllReduce", source)
        self.assertIn("device_count < 2", source)
        self.assertIn("gfx906", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_rocblas_smoke_covers_supported_fp32_fp64_tensile_path(self):
        script = (ROOT / "scripts/run_mi50_rocblas_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/rocblas/mi50_rocblas_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lrocblas", script)
        self.assertIn("/dev/kfd", script)
        self.assertIn("rocblas_sgemm", source)
        self.assertIn("rocblas_dgemm", source)
        self.assertIn("rocblas_hgemm", source)
        self.assertIn("FP16", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_rocblas_int8_smoke_isolated_validate_per_kernel_path(self):
        script = (ROOT / "scripts/run_mi50_int8_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/rocblas/mi50_rocblas_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("--int8", script)
        self.assertIn("rocblas_gemm_ex", source)
        self.assertIn("rocblas_datatype_i8_r", source)
        self.assertIn("rocblas_datatype_i32_r", source)
        self.assertIn("validate-per-kernel", source)
        self.assertIn("return 78", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_int8_dot4_smoke_covers_gcn5_packed_primitive(self):
        script = (ROOT / "scripts/run_mi50_int8_dot4_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_int8_dot4_smoke.hip").read_text(encoding="utf-8")
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        suite = (ROOT / "scripts/mi50_validation_suite.py").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("amd_mixed_dot", source)
        self.assertIn("make_char4", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("run_mi50_int8_dot4_smoke.sh", host_runner)
        self.assertIn("INT8-dot4", suite)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_int8_dot4_gemm_smoke_exercises_fallback_gemm(self):
        script = (ROOT / "scripts/run_mi50_int8_dot4_gemm_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_int8_dot4_gemm_smoke.hip").read_text(encoding="utf-8")
        host_runner = (ROOT / "scripts/run_host_tests.sh").read_text(encoding="utf-8")
        suite = (ROOT / "scripts/mi50_validation_suite.py").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("int8_dot4_gemm", source)
        self.assertIn("amd_mixed_dot", source)
        self.assertIn("k = 68", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("run_mi50_int8_dot4_gemm_smoke.sh", host_runner)
        self.assertIn("INT8-dot4-GEMM", suite)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_library_abi_smoke_covers_supported_rocm_math_stack(self):
        script = (ROOT / "scripts/run_mi50_library_abi_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_library_abi_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        for library in ("-lMIOpen", "-lrocfft", "-lrocrand", "-lrocsparse", "-lrocsolver", "-lrocblas"):
            self.assertIn(library, script)
        for symbol in (
            "miopenCreate",
            "rocfft_setup",
            "rocrand_create_generator",
            "rocsparse_create_handle",
            "rocsolver_get_version_string",
        ):
            self.assertIn(symbol, source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_miopen_smoke_exercises_gfx906_convolution_path(self):
        script = (ROOT / "scripts/run_mi50_miopen_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/miopen/mi50_miopen_convolution_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lMIOpen", script)
        self.assertIn("miopenFindConvolutionForwardAlgorithm", source)
        self.assertIn("miopenConvolutionForward", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_fft_rand_smoke_covers_native_output_correctness(self):
        script = (ROOT / "scripts/run_mi50_fft_rand_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_fft_rand_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lrocfft", script)
        self.assertIn("-lrocrand", script)
        self.assertIn("rocfft_execute", source)
        self.assertIn("rocrand_generate_uniform", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_sparse_solver_smoke_covers_native_csr_and_lu_correctness(self):
        script = (ROOT / "scripts/run_mi50_sparse_solver_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_sparse_solver_smoke.cpp").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("-lrocsparse", script)
        self.assertIn("-lrocsolver", script)
        self.assertIn("rocsparse_spmv", source)
        self.assertIn("rocsparse_v2_spmv", source)
        self.assertIn("rocSPARSE v1 fallback", source)
        self.assertIn("rocsparse_spmv_set_input", source)
        self.assertIn("rocsolver_sgetrf_npvt", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_prim_thrust_smoke_covers_wave64_reduction(self):
        script = (ROOT / "scripts/run_mi50_prim_thrust_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_prim_thrust_smoke.hip").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("rocprim/rocprim.hpp", source)
        self.assertIn("thrust::reduce", source)
        self.assertIn("rocprim::reduce", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

    def test_memory_smoke_is_bounded_and_native(self):
        script = (ROOT / "scripts/run_mi50_memory_smoke.sh").read_text(encoding="utf-8")
        source = (ROOT / "tests/hip/mi50_memory_smoke.hip").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=gfx906", script)
        self.assertIn("MI50_MEMORY_TEST_MIB", source)
        self.assertIn("hipMemGetInfo", source)
        self.assertIn("hipMemset", source)
        self.assertIn("hipMemcpy", source)
        self.assertIn("expected gfx906/wave64", source)
        self.assertIn("return 77", source)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", source)

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
        self.assertIn('EXPERIMENTAL_NEW_ISA_PORTS^^}" == "ON"', builder)
        self.assertIn("-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON", builder)
        self.assertIn("-DTHEROCK_ENABLE_HIPTENSOR=ON", builder)
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", builder)
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

    def test_llm_benchmark_scopes_requested_rocm_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            rocm_root = prefix / "rocm"
            for relative in ("bin", "lib", "lib/llvm/bin", "lib/llvm/lib", "lib/rocm_sysdeps/lib"):
                (rocm_root / relative).mkdir(parents=True, exist_ok=True)
            environment = runtime_environment(str(prefix))
            expected_root = str(rocm_root.resolve())
        self.assertEqual(environment["ROCM_PATH"], expected_root)
        self.assertTrue(environment["PATH"].startswith(str(rocm_root / "bin")))
        self.assertTrue(environment["LD_LIBRARY_PATH"].startswith(str(rocm_root / "lib")))

    def test_llm_benchmark_resolves_bare_tools_from_scoped_path(self):
        with tempfile.TemporaryDirectory() as directory:
            rocm_bin = Path(directory) / "bin"
            rocm_bin.mkdir()
            tool = rocm_bin / "llama-bench"
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o755)
            environment = {"PATH": str(rocm_bin)}
            self.assertEqual(command_path("llama-bench", environment=environment), str(tool))

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
