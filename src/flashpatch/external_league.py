"""Lossless, receipt-bound input lanes for external detector comparisons.

This module deliberately does not score or declare a winner.  It freezes one
renderer-owned RGB/CFR input, verifies that its FFV1 conversion round-trips
byte-for-byte, and records what each external executable actually consumed.
Aggregation and claim gates remain separate fail-closed steps.
"""

from __future__ import annotations

import base64
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import cv2

from . import core as flashpatch_core
from .core import analyze
from .l7_external_host import (
    VERIFICATION_SCHEMA_V2 as EXTERNAL_HOST_VERIFICATION_SCHEMA_V2,
    capture_host_identity as capture_external_host_identity,
    freeze_external_host_witness_request,
    verify_external_host_witness,
    write_external_host_witness_request,
)


class ExternalLeagueError(ValueError):
    """A comparison input or external-run receipt is not trustworthy."""


def _uv_executable() -> Path:
    """Resolve uv from the executing host instead of binding a developer home."""
    executable = shutil.which("uv")
    return Path(executable).resolve() if executable else Path("uv")


COMPARATOR_CENSUS_SCHEMA = "flashpatch-l7-comparator-census-manifest-v3"
COMPARATOR_CENSUS_RECEIPT_SCHEMA = "flashpatch-l7-comparator-census-receipt-v3"
EA_IRIS_RELEASE_ORACLE_ID = "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU"
EA_IRIS_SOURCE_ADAPTER_ID = "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC"
EA_IRIS_LEGACY_JSON_ID = "EA_IRIS_LEGACY_SOURCE_JSON_UNVERIFIED"
KAYA_DIRECT_PARTICIPANT_ID = "KAYA_PSE_DETECTION_CORRECTION_0776EA3"
DIRECT_DETECTOR_POPULATION = (
    "FlashPatch",
    KAYA_DIRECT_PARTICIPANT_ID,
    "TooFlashy",
)
CONFORMANCE_ORACLE_POPULATION = (EA_IRIS_RELEASE_ORACLE_ID,)
_CENSUS_VALIDATION_CACHE: dict[tuple[str, str], dict[str, object]] = {}
_CANONICAL_DECODE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
_KAYA_PARTICIPANT_CACHE: dict[str, dict[str, object]] = {}
EXCLUDED_SEMANTIC_MISMATCH_POPULATION = (EA_IRIS_SOURCE_ADAPTER_ID,)
MITIGATION_POPULATION = ("FFmpeg vf_photosensitivity",)
RESERVE_DETECTOR_POPULATION = ("EPI-LENS",)

FAIR_RUNTIME_PROTOCOL_SCHEMA = "flashpatch-l7-fair-runtime-protocol-v1"
FAIR_RUNTIME_RUN_SCHEMA = "flashpatch-l7-fair-runtime-run-v1"
FAIR_RUNTIME_BUNDLE_SCHEMA = "flashpatch-l7-fair-runtime-bundle-v1"
FAIR_RUNTIME_SCHEDULE_SCHEMA = "flashpatch-l7-fair-runtime-schedule-v1"
DECODER_TIMELINE_PARITY_SCHEMA = "flashpatch-l7-decoder-timeline-parity-v1"
TOOFLASHY_PARITY_ADAPTER_SCHEMA = "flashpatch-l7-tooflashy-parity-adapter-v1"
TOOFLASHY_CONFORMANCE_FIXTURES_SCHEMA = "flashpatch-l7-tooflashy-conformance-fixtures-v1"
TOOFLASHY_COPIED_REPLAY_WITNESS_SCHEMA = "flashpatch-l7-tooflashy-copied-replay-witness-v1"
EA_IRIS_SOURCE_BUILD_SCHEMA = "flashpatch-l7-ea-iris-source-build-v1"
EA_IRIS_SOURCE_BUILD_SCHEMA_V2 = "flashpatch-l7-ea-iris-source-build-v2"
EA_IRIS_SOURCE_RUN_SCHEMA = "flashpatch-ea-iris-source-frame-adapter-run-v1"
EA_IRIS_SOURCE_VIDEO_RUN_SCHEMA = "flashpatch-ea-iris-d969-source-video-conformance-run-v1"
EA_IRIS_SOURCE_CONFORMANCE_MANIFEST_SCHEMA = "flashpatch-l7-ea-iris-source-conformance-manifest-v2"
EA_IRIS_SOURCE_CONFORMANCE_RECEIPT_SCHEMA = "flashpatch-l7-ea-iris-source-conformance-receipt-v2"
EA_IRIS_REALTIME_SEMANTIC_PROBE_SCHEMA = "flashpatch-l7-ea-iris-realtime-semantic-probe-v1"
KAYA_CONFORMANCE_PROTOTYPE_SCHEMA = "flashpatch-l7-kaya-conformance-prototype-v1"
KAYA_PARTICIPANT_CONFORMANCE_SCHEMA = "flashpatch-l7-kaya-participant-conformance-v1"
KAYA_CONFORMANCE_CHILD_SCHEMA = "flashpatch-l7-kaya-conformance-child-v1"
KAYA_DIRECT_INPUT_SCHEMA = "flashpatch-l7-kaya-direct-rgb-input-v1"
KAYA_NATURAL_CASE_PARITY_SCHEMA = "flashpatch-l7-kaya-natural-case-parity-v1"
KAYA_FAIR_RUNTIME_RUN_SCHEMA = "flashpatch-l7-kaya-fair-runtime-run-v1"
KAYA_FAIR_RUNTIME_REPEATS_SCHEMA = "flashpatch-l7-kaya-fair-runtime-repeats-v1"
EXTERNAL_SLOT_CHILD_JOIN_SCHEMA = "flashpatch-l7-external-slot-child-join-v1"
KAYA_PROTOTYPE_ID = "KAYA_SOURCE_DIRECT_INPUT_PROTOTYPE_0776EA3E_UNSCORED"
KAYA_SOURCE_REVISION = "0776ea3e6949a62d5becb8027a2765961b515793"
KAYA_SOURCE_TREE = "34bef92fcc2bbd7c2e779475b84094493ae23aa1"
KAYA_REPOSITORY_URL = "https://github.com/samfatu/pse-detection-correction"
KAYA_REQUIRED_SOURCE_HASHES = {
    "LICENSE": "cd18b47f83e5cb4640272cc95e0144206be9ed55d8458573eba3cf0b49534da8",
    "requirements.txt": "da947acb06c423a8fb16487983383e98b19399f9fb585fdbe53e0c5bdc419961",
    "custom_video.py": "2704535a5c98d2f266ec71463a1d485dcaed088798ffc2fcbb2b9bc62e53630c",
    "PhotosensitivitySafetyEngine/engine/analysis.py": "397f3a3289118474708369dc338f4bc0f39e776d46af72cab8f3269d5d46cd79",
    "PhotosensitivitySafetyEngine/guidelines/w3c.py": "6b6053182bf683cca65e880c909f47f329378e462e9acb18667e659b9a0766ab",
    "PhotosensitivitySafetyEngine/libraries/common_functions.py": "b4991fbfe26fb2de44f255a3f17119123b310f710e215111857538e871678728",
    "PhotosensitivitySafetyEngine/libraries/custom_functions.py": "90fea09133824f63f7b419f187716dd906f20cd921c06e3ad78f43ba70519d77",
    "PhotosensitivitySafetyEngine/libraries/function_objects.py": "cba23e6f74bdb34de95a20a8ec09d42bcaf7ed65df2724f060b8dd7ad6e15e17",
}
KAYA_REQUIRED_DISTRIBUTIONS = {
    "cycler": "0.12.1",
    "kiwisolver": "1.5.0",
    "matplotlib": "3.5.1",
    "numpy": "1.21.2",
    "opencv-python": "4.6.0.66",
    "packaging": "26.2",
    "pillow": "12.3.0",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
}
KAYA_CALLABLE_SOURCE_HASHES = {
    "GuidelineProcess.analyse_file": "a6157a5aca4405ab35eb1a74c8d431b27a4c070f037ba2cebcc53017183a1830",
    "Display": "917eb81a7e931e22abbecdcd086c7555f17b910d212bd6530ca72733f4c7679e",
    "CustomVideo.frame_intervals": "9606b9a046de3a66659c4143f5343115a44baf3109ec8f32e482133d6ce57e02",
    "function_objects": "b52727ddc842a32eb2f13e2485daff3c4a8aff6801ae279353ffa6d311512ee1",
}
KAYA_PIPELINE_SHA256 = "f5cb1f76491fa52c1500b18aa0edb8a5ed1ebd0b7b2f0ce4548f29f0c26ca391"
KAYA_PYTHON_SHA256 = "685193c2432feb9d2a2b3ba129d976a7fd4172c60e1622b7b7ffd4308c40f6a5"
KAYA_COMMON_BASE_RUNTIME_CLOSURE = {
    "classification": "PINNED_IMMUTABLE_RUNTIME_BASE_REQUIRES_STORAGE_INDEPENDENCE",
    "entry_count": 5422,
    "content_bytes": 80598705,
    "tree_sha256": "b2af88a9ea833e90a14d4769b1496c5d2c6332266231aa75b104afc75fd90608",
}
KAYA_IMPORT_HOOKS = {
    "meta_path": [
        ["_frozen_importlib", "BuiltinImporter"],
        ["_frozen_importlib", "FrozenImporter"],
        ["_frozen_importlib_external", "PathFinder"],
        ["six", "_SixMetaPathImporter"],
    ],
    "path_hooks": [
        ["zipimport", "zipimporter"],
        ["_frozen_importlib_external", "FileFinder.path_hook.<locals>.path_hook_for_FileFinder"],
    ],
}
KAYA_DISTRIBUTION_CLOSURES = {
    "cycler": {
        "normalized_record_sha256": "d336a450b93e33780693fa1ea208cbde9730d7cbfd98126e7743f2c5fa9af7c3",
        "portable_file_count": 8,
        "portable_files_sha256": "fb211d31b312a16478679126390102b80da524df05568d2510a5e62b27ca8a4d",
        "module_path": "lib/python3.10/site-packages/cycler/__init__.py",
        "module_sha256": "d4975182fe59cf1a3e5b57afec1fcb5aabb2b163fa2d91fa08793f08eb486971",
    },
    "kiwisolver": {
        "normalized_record_sha256": "5de1e73f73d6d2bb76d55a184db678c4f3fb820c1f28f2e35f865fdb825305c3",
        "portable_file_count": 11,
        "portable_files_sha256": "9e81212651d69c023a14030440caa70bd94332bc84c957a10d677a9b32a63819",
        "module_path": "lib/python3.10/site-packages/kiwisolver/__init__.py",
        "module_sha256": "0d42d2b355c4b47ac2c8b84dfe0f89f0f8af557b6f4b128dd97cf98116ca9122",
    },
    "matplotlib": {
        "normalized_record_sha256": "78d5c8b0244202a657b54e8bc8c35a148c87f159ef9e90090842b25743669320",
        "portable_file_count": 529,
        "portable_files_sha256": "3170b6596c2110656127f321ab9cca821f205d817500b03bbb8c535f97e5884a",
        "module_path": "lib/python3.10/site-packages/matplotlib/__init__.py",
        "module_sha256": "a19a045ba96b038b19844118d2e4e8aa281e5141398658b21376d9e0e7abe300",
    },
    "numpy": {
        "normalized_record_sha256": "7b65d299b0991f60aa703fadeeeeb64f143c6685f14c3c95179ea921b61bb088",
        "portable_file_count": 712,
        "portable_files_sha256": "16537179a9db995d2ae0b5b18120287ed4907120f5df41c2dc4ba53fd11ccef4",
        "module_path": "lib/python3.10/site-packages/numpy/__init__.py",
        "module_sha256": "02e54b39a5825a5eb95c74639b22593a8af816212f9bb1fdcb78f17112a5d27f",
    },
    "opencv-python": {
        "normalized_record_sha256": "752f90038f16bcf832cbd3dca3c34a56dad78f52049639427e09e4e275340861",
        "portable_file_count": 84,
        "portable_files_sha256": "8c6fa5be88fdacb5b57584b65903694fcf4745275386feed851ce9363818a15e",
        "module_path": "lib/python3.10/site-packages/cv2/__init__.py",
        "module_sha256": "3cc4a7613da6793421bb59e01d77148b03d37a774e44579c03d1c37407f0ebbe",
    },
    "packaging": {
        "normalized_record_sha256": "0291ff961cf9e2732b13da0c8cd8b2aa94d4935dd5275e5fce26d59c7de3e58e",
        "portable_file_count": 28,
        "portable_files_sha256": "a7a803628b4a59de7bc7d16cad012f7f131934566def906c5b4f092491cc6b2c",
        "module_path": "lib/python3.10/site-packages/packaging/__init__.py",
        "module_sha256": "42130474fbb65e882b2735774b42964bab7b97423d93c11e0d1265e1f9f0f3bb",
    },
    "pillow": {
        "normalized_record_sha256": "f0629a1d141a8be7b054b7c14a622d151b91ca3e8f212d4f61d3e6b78f69fb3d",
        "portable_file_count": 141,
        "portable_files_sha256": "e8ed90ad1888ea9010a5a21aafc7856c736c8c0c705461c698c1e1cd8747c9a8",
        "module_path": "lib/python3.10/site-packages/PIL/__init__.py",
        "module_sha256": "7361b6ad3878589affe5956e4a4da24d71398b0891fd3a8c15e9362217b4ca01",
    },
    "pyparsing": {
        "normalized_record_sha256": "3a25bda3ddedd53a3a3f97a924fb19753ada3145c233d640e564c7991e154dfc",
        "portable_file_count": 24,
        "portable_files_sha256": "6065751d0d53a20bf60b4d6d7273aceb2e5f0083b784f7d4aac5a4b2b9b2cd41",
        "module_path": "lib/python3.10/site-packages/pyparsing/__init__.py",
        "module_sha256": "08c1cce44cc9489825cfe3c2012bff9395b5e2d46486b09fe86488e0aacf2f6e",
    },
    "python-dateutil": {
        "normalized_record_sha256": "3f74cb322a1be616f4302ba3d7bcdd4f73119545a8905b74d939ec35f810977c",
        "portable_file_count": 26,
        "portable_files_sha256": "19c57a119691ea1520c41786da4e2bf3838364a6d84627c882826cffb23bc68c",
        "module_path": "lib/python3.10/site-packages/dateutil/__init__.py",
        "module_sha256": "32a6a6ebb58ef4891399417223aeaf4ba2284974e9f46dfcf0369d1f62c230b6",
    },
    "six": {
        "normalized_record_sha256": "8e98d2d8310169ea395e9085f3c471323f478ed9377ebad2a41f56331df15675",
        "portable_file_count": 7,
        "portable_files_sha256": "fc9794407ab44b916d36ea7022d0f5e22f12706d4255b61ab5b23b8850ca5885",
        "module_path": "lib/python3.10/site-packages/six.py",
        "module_sha256": "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
    },
}
KAYA_REQUIRED_FIXTURE_IDS = (
    "safe",
    "rgb-channel-trap",
    "flash-threshold",
    "history-59",
    "history-60",
    "history-61",
    "letterbox",
    "state-reuse",
)
EA_IRIS_SOURCE_REVISION = "d96978ac1107f3463b77f69a9c1b1ec5d45291a0"
EA_IRIS_SOURCE_TREE = "788043e2673b9764abe8a431567a74213325ad2b"
EA_IRIS_SOURCE_VIDEO_ORACLE_ID = "EA_IRIS_SOURCE_VIDEO_PATH_D96978AC_CONFORMANCE_ONLY"
EA_IRIS_SOURCE_CONFIG_SHA256 = "9b9bbaec85c3c0a04a16d12e1c71299fe58797c42f4b24a04f6489d69b541572"
EA_IRIS_SOURCE_LICENSE_SHA256 = "bf08b9f24b0e55e96edceeeb20f0ee077ab486c5fc2acbaa5e81d42aa9f207cf"
EA_IRIS_SOURCE_EXAMPLE_SHA256 = "84efd3b1306f0d8b18e57193f77b7fd0e429da73e54463267651ebccd1948b55"
EA_IRIS_SOURCE_VCPKG_MANIFEST_SHA256 = "f8ed5f39987dcdce4662c1531bba9e6775b482a457351a85adc44cc79e992c6a"
EA_IRIS_MINIMAL_MEDIA_SCHEMA = "flashpatch-l7-ea-iris-minimal-media-toolchain-v1"
EA_IRIS_MINIMAL_OPENCV_SOURCE = {
    "tag": "4.8.0",
    "tag_object": "53296de62872b5e7d042ddffb49679fbdcca99f6",
    "revision": "f9a59f2592993d3dcc080e495f4f5e02dd8ec7ef",
    "tree": "59447338691243882d49a471b32bed0d20d1f1b5",
    "archive_sha256": "cbf47ecc336d2bff36b0dcd7d6c179a9bb59e805136af6b9670ca944aef889bd",
    "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "license": "Apache-2.0",
}
EA_IRIS_MINIMAL_FFMPEG_SOURCE = {
    "tag": "n6.1.1",
    "tag_object": "6f4048827982a8f48f71f551a6e1ed2362816eec",
    "revision": "e38092ef9395d7049f871ef4d5411eb410e283e0",
    "tree": "dc2cc148dbd7406686d6052c6d414594ad9c5c20",
    "archive_sha256": "7c1ebea95d815e49c1e60c7ee816410dec73a81b8ac002b276780d2f9048e598",
    "license_sha256": "b634ab5640e258563c536e658cad87080553df6f34f62269a21d554844e58bfe",
    "license": "LGPL-2.1-or-later",
}
EA_IRIS_MINIMAL_OPENCV_MODULES = (
    "opencv_core",
    "opencv_imgproc",
    "opencv_features2d",
    "opencv_imgcodecs",
    "opencv_videoio",
)
EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS = {
    "src/PatternDetection.cpp": {
        "sha256": "f46e1d258ebed2b352b53c644acbfdf5a60d99583ce8d9521d79f61f77d84a06",
        "required_header": "#include <opencv2/features2d.hpp>",
        "required_module": "opencv_features2d",
    },
    "example/main.cpp": {
        "sha256": EA_IRIS_SOURCE_EXAMPLE_SHA256,
        "required_runtime_path": "iris::VideoAnalyser::AnalyseVideo",
        "required_module": "opencv_videoio",
    },
}
EA_IRIS_MINIMAL_FFMPEG_SOURCE_REQUIREMENTS = {
    "src/VideoAnalyser.cpp": {
        "sha256": "6f4c52c538446b1588961a4dc58b689b56e41feb389e176467edc31d592f46e6",
        "required_headers": (
            "#include <libavformat/avformat.h>",
            "#include <libavcodec/avcodec.h>",
        ),
        "required_calls": (
            "avformat_alloc_context",
            "avformat_open_input",
            "avformat_find_stream_info",
            "avformat_close_input",
        ),
    },
}
EA_IRIS_MINIMAL_OPENCV_FORBIDDEN_CACHE = {
    "BUILD_opencv_apps": "OFF",
    "BUILD_opencv_java": "OFF",
    "BUILD_opencv_js": "OFF",
    "BUILD_opencv_perf_tests": "OFF",
    "BUILD_opencv_python2": "OFF",
    "BUILD_opencv_python3": "OFF",
    "BUILD_opencv_tests": "OFF",
    "BUILD_opencv_world": "OFF",
    "WITH_1394": "OFF",
    "WITH_CUDA": "OFF",
    "WITH_EIGEN": "OFF",
    "WITH_GDAL": "OFF",
    "WITH_GDCM": "OFF",
    "WITH_GSTREAMER": "OFF",
    "WITH_GTK": "OFF",
    "WITH_IPP": "OFF",
    "WITH_LAPACK": "OFF",
    "WITH_OPENCL": "OFF",
    "WITH_OPENEXR": "OFF",
    "WITH_OPENGL": "OFF",
    "WITH_OPENMP": "OFF",
    "WITH_QT": "OFF",
    "WITH_TBB": "OFF",
    "WITH_V4L": "OFF",
}
EA_IRIS_MINIMAL_FFMPEG_COMPONENTS = {
    "libraries": ("avcodec", "avformat", "avutil", "swscale"),
    "decoders": ("ffv1",),
    "demuxers": ("matroska",),
    "protocols": ("file",),
}
EA_IRIS_REQUIRED_CONFORMANCE_ROLES = (
    "SAFE_CONTROL",
    "LUMINANCE_FLASH",
    "RED_FLASH",
    "PATTERN",
)
EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES = {
    "PATTERN_PERSISTENCE_FRAMES": (29, 30, 31),
    "ONE_SECOND_FLASH_FRAMES": (59, 60, 61),
    "FOUR_SECOND_EXTENDED_FRAMES": (239, 240, 241),
    "FIVE_SECOND_WINDOW_FRAMES": (299, 300, 301),
    "TRANSITION_COUNT": (3, 4, 6, 7),
}
EA_IRIS_PATTERN_NEGATIVE_FIXTURE = {
    "path": "test/Iris.Tests/data/TestImages/Patterns/20stripes.png",
    "sha256": "c98fe3118152dc909f94b169c5f718c71d65380781af77925753ed91d60a7c5f",
    "transform": "cv2-4.13.0-inter-area-640x360-bgr-to-rgb-repeat-31",
    "expected_rgb_frame_sha256": "db17d9862b4d2d43a66ca174af46e88bc89b0eec6f6f67b58ab16acff0cafaea",
    "frame_count": 31,
    "fps": 60,
}
EA_IRIS_MINIMAL_SUPPORT_CLOSURE = {
    "libfmt-dev": ("9.1.0+ds1-2", "amd64", "cc05cae4b7c6e541b6871e3333329605c0d90312a852f2c3b507ad5c30dd9914"),
    "libfmt9": ("9.1.0+ds1-2", "amd64", "0a9dd337aea7ae59aba44994a92dd2780365b534a714518477c4e906f0efd297"),
    "libpkgconf3": ("1.8.1-2build1", "amd64", "fc3a57e8f931ec06cb7c51acc56878f1fec247ff02a7adcab26905c2faeb2792"),
    "libspdlog-dev": ("1:1.12.0+ds-2build1", "amd64", "850b97a93e252c1d2f9d8e9f59cac919015efb56241138098637f84a188ae3a0"),
    "libspdlog1.12": ("1:1.12.0+ds-2build1", "amd64", "e69e315e1596b6cf97a714ae797e55d42765664a739a0b30beac35164a4edaff"),
    "libssl-dev": ("3.0.13-0ubuntu3.12", "amd64", "9a5cf7bc8e876ef4498ddf0180b6fafe0e52c2a8da2f06f8bc78c2a6fc92ec58"),
    "libssl3t64": ("3.0.13-0ubuntu3.12", "amd64", "6a963adb1106fca567d24d4a1e5da0bad25de79ac2564cd1ba846e677e1c951b"),
    "ninja-build": ("1.11.1-2", "amd64", "6a17f76a0f75586f1292f6c61a9b07fd0dec963baf6a102b17673f601c6b323a"),
    "nlohmann-json3-dev": ("3.11.3-1", "all", "85e4e95dc4bdc6034d593e1933ee88e085319fee36cae83c178e8ddc1a66e8fd"),
    "pkgconf": ("1.8.1-2build1", "amd64", "834a58031069d97d7cfb8b2f5bfd5effc69cecf7f30cc362071875f1f8dc1828"),
    "pkgconf-bin": ("1.8.1-2build1", "amd64", "7a812f05ee1610154b433e2ad54f6e4163fcbb306b9fb31afe959afb2e5e1545"),
}
EA_IRIS_MINIMAL_BUILD_TOOL_CLOSURE = {
    "bash": ("5.2.21-2ubuntu4", "amd64", "73de311a21e094e29ac01527d2b52226cc87fde0a5b57032902251b426d92c66"),
    "binutils": ("2.42-4ubuntu2.8", "amd64", "151dcf94179a8a3264fb3fc2e3fbab8cf5021e557d40f7d6a115d496d225b16d"),
    "binutils-common": ("2.42-4ubuntu2.8", "amd64", "3db2f9cd5488ee608a8745a1afe729cf4aaefa41c0d16a4671928ee53936e8ad"),
    "binutils-x86-64-linux-gnu": ("2.42-4ubuntu2.8", "amd64", "6503593d06ecd28e1e6ab061cc9eea39affd28e34c72fbc19142525ad313d838"),
    "cmake": ("3.28.3-1build7", "amd64", "4b0a7f8c0daf27b26b46997d994ae5d1ee7a3d11dfcda9f7627bb1462c162295"),
    "cmake-data": ("3.28.3-1build7", "all", "20a3b644211ce82f35c24f1f5052199adeb0ac159978a8740fd6c0959611557f"),
    "cpp-13-x86-64-linux-gnu": ("13.3.0-6ubuntu2~24.04.1", "amd64", "2ca48bf0c2d6465bc39322899715a85d934b4d7442dd5586a7bebbe3ce0f806b"),
    "dash": ("0.5.12-6ubuntu5", "amd64", "e97728d8deaa51300255f0572bbd68b9549e0894a184c056dc420fc4e0ba0781"),
    "g++-13-x86-64-linux-gnu": ("13.3.0-6ubuntu2~24.04.1", "amd64", "0bd6af6164252d4ea9170d201ee4c10a3120fe0fc04985a00be3bc9076353844"),
    "gcc-13-x86-64-linux-gnu": ("13.3.0-6ubuntu2~24.04.1", "amd64", "a134b0319a82d14581b3a14820d2832af4ec9778ed8b9b4ddaeecfb0555ec325"),
    "grep": ("3.11-4build1", "amd64", "fc0fdc5983ea3d3579ccf335e51dec69684a0dd9bb915734999c5733add9507a"),
    "libbinutils": ("2.42-4ubuntu2.8", "amd64", "16b5f6c672295ed855bf1945010971f8b483276b134bbdab9e32a75321b21f53"),
    "libc-dev-bin": ("2.39-0ubuntu8.8", "amd64", "c894e5a5f137429657d09e853fbbb19d53fc164c60804c396cd43873b0b4f734"),
    "libc6-dev": ("2.39-0ubuntu8.8", "amd64", "bb8741966e7c1d2e2c0b84bb311717a0908fb563d9b2247b7212710e3cd88b94"),
    "libgcc-13-dev": ("13.3.0-6ubuntu2~24.04.1", "amd64", "cd689db2691edaa10f37329307292796bb599e722e0505c79e14caaa1fe9a93a"),
    "libstdc++-13-dev": ("13.3.0-6ubuntu2~24.04.1", "amd64", "ee5633e863e19c3381ed97842ce35ed32ede96a3d1ae4e94c051d3036fe21347"),
    "linux-libc-dev": ("6.8.0-136.136", "amd64", "adde3ef425ccf82bf5f14d45a3143fab6c6eea26c1fd0f8a5eb43989e5fe255d"),
    "make": ("4.3-4.1build2", "amd64", "1fe6a815b56c7b6e9ce4086a363f09444bbd0a0d30e230c453d0b78e44b57a99"),
    "mawk": ("1.3.4.20240123-1build1", "amd64", "dc7f7f4dad4b48f6012ea65de3198d8376604afef39f06d65ec6167740e203c9"),
    "perl-base": ("5.38.2-3.2ubuntu0.3", "amd64", "d6032f2e6e065dfa8785ec913bb2faf9019fff51965ae6f15d7a7aadf8f5ab3f"),
    "python3.12-minimal": ("3.12.3-1ubuntu0.15", "amd64", "487383dc2a895e0a767d820e0e55f2ab7d6ebe4dccd3d2c0b81f00ee11bb1152"),
    "sed": ("4.9-2ubuntu0.24.04.1", "amd64", "5406c0950df2653be3d4acaf9f169c51d30cf4a617b63423e64d2853b0d72b5c"),
}
EA_IRIS_SOURCE_CPP_PATHS = (
    "src/Configuration.cpp",
    "src/Flash.cpp",
    "src/FlashDetection.cpp",
    "src/FpsFrameManager.cpp",
    "src/FrameRgbConverter.cpp",
    "src/Log.cpp",
    "src/PatternDetection.cpp",
    "src/RedSaturation.cpp",
    "src/RelativeLuminance.cpp",
    "src/ScopeProfiler.cpp",
    "src/TimeFrameManager.cpp",
    "src/TransitionTracker.cpp",
    "src/VideoAnalyser.cpp",
    "utils/src/BaseLog.cpp",
    "utils/src/FrameConverter.cpp",
)
EA_IRIS_SOURCE_BOUNDARY_METHODS = (
    "iris::Configuration::Init",
    "iris::VideoAnalyser::RealTimeInit",
    "iris::VideoAnalyser::AnalyseFrame",
    "iris::VideoAnalyser::DeInit",
)
EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN = {
    "compiler": ("/usr/bin/x86_64-linux-gnu-g++-13", "1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769"),
    "archiver": ("/usr/bin/x86_64-linux-gnu-ar", "534681ac11c18868cfc4fdf98770aa0ba8973eedc90c231e94e6ba96e1a04f27"),
    "dpkg_deb": ("/usr/bin/dpkg-deb", "fc9dde782ca8309b65ab92e61d13318bf463cd1aa71daa8967b87b81b48b63d2"),
    "readelf": ("/usr/bin/x86_64-linux-gnu-readelf", "871be389739ecf9924b052c2fde4d2a2068a54e882201b9c34897337a5a0a130"),
    "ldd": ("/usr/bin/ldd", "4f1d37e25f27535e3f02a5b7da63e1ce18d4982445db2c25fc8f985a3d395cc3"),
    "git": ("/usr/bin/git", "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"),
}
EA_IRIS_SOURCE_DEBIAN_CLOSURE = {
    "libaec0": ("1.1.2-1build1", "amd64", "c045fd79f37e38e69520172137700cee2dd2be72fa2ce5698a3a835809bfd2e0"),
    "libarmadillo12": ("1:12.6.7+dfsg-1build2", "amd64", "fa5d9106268cf8ef3a513764252baa9c1f413a5b104a813be46a5a094efd4577"),
    "libarpack2t64": ("3.9.1-1.1build2", "amd64", "b9d203dd5fbbd8ce353c9ac366d02fe029f4e59d9e29554d8f63ef5674d62519"),
    "libavcodec-dev": ("7:6.1.1-3ubuntu5", "amd64", "4b54c484f58793f7b223763f764a01d558a3707d16ac6dd804c754f430eff369"),
    "libavcodec60": ("7:6.1.1-3ubuntu5", "amd64", "970d92c1697f762235af439d72df8c6cb492be45050a1a00bbe0d0842a137f93"),
    "libavformat-dev": ("7:6.1.1-3ubuntu5", "amd64", "073317ed5657bb937a55bbdba6a6e651448396b985efa97382d1363294a57777"),
    "libavformat60": ("7:6.1.1-3ubuntu5", "amd64", "3a855488d50b4ebe04425e5f690e00ebd65d70e60f85c9765d5823a2f82b4969"),
    "libavutil-dev": ("7:6.1.1-3ubuntu5", "amd64", "3f43e8fd0353fda23ab1051ad2c99e137025f6b4bed9a4c0226139d5b6a871c8"),
    "libavutil58": ("7:6.1.1-3ubuntu5", "amd64", "e57f8cc358f4b1b2af721ca6242723dc12290df543a142811ce7747e2d5b30cc"),
    "libblosc1": ("1.21.5+ds-1build1", "amd64", "b88202d54ada996226e5779717b886589134cfc53db5cd982c8cfcc0eabbf1c9"),
    "libcfitsio10t64": ("4.3.1-1.1build2", "amd64", "2ff680293eebb0977554f126ee62910c7caf172abcab7d7e7e51e37a5e4b1654"),
    "libcharls2": ("2.4.2-2build2", "amd64", "4e0899b98fe57b5011412bfdb2fa96f2559c746f05bb1604da306642940ceaec"),
    "libfmt-dev": ("9.1.0+ds1-2", "amd64", "cc05cae4b7c6e541b6871e3333329605c0d90312a852f2c3b507ad5c30dd9914"),
    "libfmt9": ("9.1.0+ds1-2", "amd64", "0a9dd337aea7ae59aba44994a92dd2780365b534a714518477c4e906f0efd297"),
    "libfreexl1": ("2.0.0-1build2", "amd64", "11a7d74c54fbef3dc2b8a87f36850f678958e4496f820ac0e56e57aff7d38a84"),
    "libfyba0t64": ("4.1.1-11build1", "amd64", "82bfd63bf3fc5849f1dc10ea3cc8b03bc677dd8201eaa76c1c8f4163009a67ec"),
    "libgdal34t64": ("3.8.4+dfsg-3ubuntu3", "amd64", "febc53aa73c8f0e9234ecbefb0f2bd93d7fa34aee8c7dde65fbc2405ac1a90de"),
    "libgdcm3.0t64": ("3.0.22-2.1ubuntu1", "amd64", "eeece84e5a5e9117991bffcd45acba349729b5a15b762fa27c566f5e65be8882"),
    "libgeos-c1t64": ("3.12.1-3build1", "amd64", "a44761e42d8434c6140759629739148f2d6dd34de74af809404fd9becbba3b1d"),
    "libgeos3.12.1t64": ("3.12.1-3build1", "amd64", "3040bab330cb02bfad385b47a448d4c4165669a928fbe19921548d33112350d7"),
    "libgeotiff5": ("1.7.1-5build1", "amd64", "7823e460621a925c08a38b5b68940d8b60eb7ca70cab7eb2da81e1168f1bda77"),
    "libhdf4-0-alt": ("4.2.16-4build1", "amd64", "a71e977335b2e3e196773ce9c1daafc89e677390cf50fbbc762ed6b0ef6ef598"),
    "libhdf5-103-1t64": ("1.10.10+repack-3.1ubuntu4", "amd64", "875e7de8aff8e6af3df2ed8b4455f767c3e9b021652a1ee2bd924f76932d22d1"),
    "libhdf5-hl-100t64": ("1.10.10+repack-3.1ubuntu4", "amd64", "ff89c6606df070ab1e4adbd9c66c46820acaed39ab61bca5e3ca947feb05fd9d"),
    "libimath-3-1-29t64": ("3.1.9-3.1ubuntu2", "amd64", "7b62292db74bab20ccea4db0039b5c7e720aec36aeed4a405d8da4c2c60b29c8"),
    "libkmlbase1t64": ("1.3.0-12build1", "amd64", "e78b8f18fd506218b0ef62c726be76d6496987f94c09ec2da721a340f90581d0"),
    "libkmldom1t64": ("1.3.0-12build1", "amd64", "ca362cb220134702357de184ae29e4eb2b6b71e813249e48d39d899b01e06baa"),
    "libkmlengine1t64": ("1.3.0-12build1", "amd64", "c31e066a9076692654dce24e407f6e6a4143ecfcc7b2413b3880e81a17c7b9f5"),
    "libminizip1t64": ("1:1.3.dfsg-3.1ubuntu2.1", "amd64", "1631892d189552939b6717aef16d126f5d0895f58a8178b070d0c59a464a11a9"),
    "libmysqlclient21": ("8.0.46-0ubuntu0.24.04.3", "amd64", "f804a4c711554f3da797d329ffa437a139fab51f59d0708879814da5df8a9ccd"),
    "libnetcdf19t64": ("1:4.9.2-5ubuntu4", "amd64", "cd4aa0b558575d7c63b46c865ce71695cf2a036475e706931fcc01390c4e64f9"),
    "libodbcinst2": ("2.3.12-1ubuntu0.24.04.1", "amd64", "020551be53bb04bd7b87b97077432793fe7bce268eca9e4b3a4fc4a3d34345d6"),
    "libogdi4.1": ("4.1.1+ds-3build1", "amd64", "7f5764f9f1a5a7031b6a7c9399f53e33973693cfa4977247ce7848951d2ae3df"),
    "libopencv-core-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "7d75d19f2f88016f8272aca044f593d0c7932f9e8485be08dbe0973edc7797e9"),
    "libopencv-core406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "22c2c50541f11f02ffedde3cb3047a2476b8afae7662c559204e9b02efb7b17c"),
    "libopencv-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "2971134fdef2d3832eccb89e47ab054d5284a9602fc329ab418e0d1f1c6aaf74"),
    "libopencv-features2d-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "4e1a5c389f1a3383b6921951c9068d1e1800302bdf6bd84a733353a465e09560"),
    "libopencv-features2d406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "ae156f6d551a6666a844b01ea4076208509251cdd7061a373e4f581b5052b267"),
    "libopencv-flann-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "14c1f6f912df1319da1d92f1c4d7f42d1a6dceb0ac7a27cfb2218d4b8e547f79"),
    "libopencv-flann406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "5d95da197ac7085a1596a74615ac84f6e29a67e92e95efa68868355d73c0fb63"),
    "libopencv-highgui-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "0a529ba4dc08ed9bc46c595bf7db3e7dbb0af2d36edda0c17d11d07f0ec8eb6b"),
    "libopencv-highgui406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "c6bf7f24dc529f6090b943f369232aa06f3549d8dcc39db518619da3d224ee57"),
    "libopencv-imgcodecs-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "6381800f5cb38413e7d2cd4094cc642858e917b5f5ad80140450ff9bcce37b99"),
    "libopencv-imgcodecs406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "66fe816e7030b5120e49c4f1ce9a2a747793be7c36f2697b640dfa23979bce5f"),
    "libopencv-imgproc-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "fca9c2836aa990ff4ad7cd9b0fcbd260d0f8b300e81dc61b99e320d854be6ea9"),
    "libopencv-imgproc406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "406b188e044ae15d98746cd1baae4e9085344f40ccb89dc141dfcb6819f6a71d"),
    "libopencv-ml-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "e79fba66e88df9607308731d61388674ce90687348f85cf7116ca245eab6862b"),
    "libopencv-ml406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "85217b993b5b206c1d57e9790dd6556d4eed98038d485688f3d8b610c8b66f97"),
    "libopencv-videoio-dev": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "c8503eefe7535831a4c673b7fe4ef94ef41b65d77d02bccdabd8a3260ec7cb3c"),
    "libopencv-videoio406t64": ("4.6.0+dfsg-13.1ubuntu1", "amd64", "2e1c08d94b186c8e9a6255c502cdc624bb0fd229649bcb67f98e4a32c5e3d819"),
    "libopenexr-3-1-30": ("3.1.5-5.1build3", "amd64", "646d78a860e71d02414df08644f70125e670fb90e8f5d834a7cefd732a9eadfe"),
    "libproj25": ("9.4.0-1build2", "amd64", "d98af14c3b9db02e4216437005a36f4ecdf52b2935038f380f63969164216410"),
    "libqhull-r8.0": ("2020.2-6build1", "amd64", "5b569643ef0cf284ccd7b81cd1bcbce820dd33deb02b1edede5287fd24c53bec"),
    "librttopo1": ("1.1.0-3build2", "amd64", "975531819057f598ceaeb6561a9d9dbc48ed2e3b0f64a7d3fa2b81c1c9600a8d"),
    "libspdlog-dev": ("1:1.12.0+ds-2build1", "amd64", "850b97a93e252c1d2f9d8e9f59cac919015efb56241138098637f84a188ae3a0"),
    "libspdlog1.12": ("1:1.12.0+ds-2build1", "amd64", "e69e315e1596b6cf97a714ae797e55d42765664a739a0b30beac35164a4edaff"),
    "libspatialite8t64": ("5.1.0-3build1", "amd64", "58ad79e7e81a49aed70e565c78ba348d0004df01917e9ffd78cf402605cd32e7"),
    "libssl-dev": ("3.0.13-0ubuntu3.12", "amd64", "9a5cf7bc8e876ef4498ddf0180b6fafe0e52c2a8da2f06f8bc78c2a6fc92ec58"),
    "libswscale-dev": ("7:6.1.1-3ubuntu5", "amd64", "34397042ccaf8ffcfd5ae09b66671dc74582d955b162783de5466a9deefc683b"),
    "libswscale7": ("7:6.1.1-3ubuntu5", "amd64", "2c17ae58b35112aace5179d7deea51faa20e82e04847b257f339376b11744e22"),
    "libsuperlu6": ("6.0.1+dfsg1-1build1", "amd64", "f4aa8ba3ccbac5595c5e901e1ec1d07f67f792937990247333bbc64f79b049cb"),
    "libsz2": ("1.1.2-1build1", "amd64", "d442719f0f7597886df1d87fb4978f15941c468a452bba7f85fd7b2d93f1400f"),
    "libtbb12": ("2021.11.0-2ubuntu2", "amd64", "78e2c79f5749fc5c55b32f9745e07d5c3480be574fd4d64bb0325cbec4206b06"),
    "liburiparser1": ("0.9.7+dfsg-2build1", "amd64", "cfbe2c305bacb90c0709a989e997cb2a1a362eebde9dc22b95c254bfa9297c57"),
    "libxerces-c3.2t64": ("3.2.4+debian-1.2ubuntu2", "amd64", "ad067537a0589df229f4ed4122d781f3e869121cef38062e3b271f4b7dd8a7b4"),
    "ninja-build": ("1.11.1-2", "amd64", "6a17f76a0f75586f1292f6c61a9b07fd0dec963baf6a102b17673f601c6b323a"),
    "nlohmann-json3-dev": ("3.11.3-1", "all", "85e4e95dc4bdc6034d593e1933ee88e085319fee36cae83c178e8ddc1a66e8fd"),
    "pkgconf": ("1.8.1-2build1", "amd64", "834a58031069d97d7cfb8b2f5bfd5effc69cecf7f30cc362071875f1f8dc1828"),
}
TOOFLASHY_PARITY_ADAPTER_REVISION = "8274e1ea09bd6099d384056f0fcb6fbc32cf0e3f"
TOOFLASHY_PARITY_ADAPTER_TREE = "665aa33c03be007ccbf485a16ce945602c22bcf6"
TOOFLASHY_OLDFILM_CANONICAL_SHA256 = "b8eef739fa17170f19b523f1bb99fba373f9a5682ead5228b566a6f435c14101"
TOOFLASHY_PARITY_SOURCE_HASHES = {
    "src/tooflashy/video.py": "23a774f1576d4b1206301b2dd2a2760dce83d0c46f0e777ce5a6f39d8e9eca9e",
    "src/tooflashy/analysis.py": "4b1d48c225c9e34c0746a2623a1bc0fd9aefddef9361f7b9aec83366db87a28a",
    "src/tooflashy/cli.py": "2508affae66a3133085be447733761b228e89252e682b4015a769b533cf91802",
    "pyproject.toml": "6a3b83e49e923484595cf042c9a5db2c6b6f4ce5c3be2cc4cc18390ec949efe6",
    "uv.lock": "2cf5f59df97177858f32a7d772c2199d9d8c8633205dbeac5c93c7697cfd18a4",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
TOOFLASHY_PARITY_CALLABLE_HASHES = {
    "iter_video_frames": "c3cf17520192efb461c69624965e373ed096fcbe68c15ebdf14ee16d468f529c",
    "analyze_frames": "09b4e9286c97835dbbf972f14927c1cf5bf8173c1fd0960008aa78133cd417dc",
}
TOOFLASHY_EDITABLE_SEMANTIC_FILES = {
    "INSTALLER": b"uv",
    "METADATA": (
        b"Metadata-Version: 2.3\n"
        b"Name: tooflashy\n"
        b"Version: 0.1.0\n"
        b"Author: orb\n"
        b"Author-email: orb <orb@orb.local>\n"
        b"Requires-Dist: numpy>=1.26\n"
        b"Requires-Dist: opencv-python>=4.9\n"
        b"Requires-Python: >=3.11\n"
    ),
    "REQUESTED": b"",
    "WHEEL": (
        b"Wheel-Version: 1.0\n"
        b"Generator: uv 0.11.12\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    ),
    "entry_points.txt": (
        b"[console_scripts]\n"
        b"tooflashy = tooflashy.cli:main\n\n"
    ),
    "uv_build.json": b"{}",
}
TOOFLASHY_EDITABLE_CONSOLE_BODY = (
    b"# -*- coding: utf-8 -*-\n"
    b"import sys\n"
    b"from tooflashy.cli import main\n"
    b"if __name__ == \"__main__\":\n"
    b"    if sys.argv[0].endswith(\"-script.pyw\"):\n"
    b"        sys.argv[0] = sys.argv[0][:-11]\n"
    b"    elif sys.argv[0].endswith(\".exe\"):\n"
    b"        sys.argv[0] = sys.argv[0][:-4]\n"
    b"    sys.exit(main())\n"
)
TOOFLASHY_OFFICIAL_JSON_FIELDS = {
    "path",
    "passes",
    "fps",
    "frame_count",
    "event_count",
    "failures",
}
EA_IRIS_RELEASE_TIMESTAMP_PRECISION_US = 1_000
FAIR_RUNTIME_BOUNDARY = {
    "id": "canonical-input-through-normalized-observation-v1",
    "start": "before_canonical_frozen_input_materialization_or_load",
    "stages": [
        "canonical_frozen_input_materialization_or_load_or_decode",
        "process_or_api_invocation",
        "native_output_parse_and_normalization",
    ],
    "end": "after_native_output_parse_and_normalization",
}
FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY = {
    "schema": "flashpatch-l7-effective-environment-policy-v1",
    "compared_fields": [
        "process_environment",
        "machine",
        "cpu.model",
        "cpu.logical_count",
        "cpu.affinity",
        "gpu.visible_device_nodes",
        "cache.policy",
        "cache.input_sha256",
        "cache.input_bytes",
    ],
    "excluded_environment_keys": [
        "PWD",
        "UV_PROJECT",
        "FLASHPATCH_L7_SCHEDULE_SHA256",
        "FLASHPATCH_L7_SCHEDULE_SLOT",
        "FLASHPATCH_L7_SCHEDULE_ROUND",
        "FLASHPATCH_L7_SCHEDULE_POSITION",
        "FLASHPATCH_L7_SCHEDULE_COMPARATOR",
        "FLASHPATCH_L7_SCHEDULE_REPEAT",
    ],
    "full_environment_treatment": "COMPARE_AFTER_DECLARED_IDENTITY_AND_SCHEDULE_EXCLUSIONS",
    "reason": "all non-identity process environment fields can affect resources or runtime behavior",
}
_RUNTIME_PROBE_SCRIPT = """
import hashlib
import json
import os
import platform
import stat
import socket
import subprocess
import sys
import time

resource_keys = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
environment = dict(os.environ)
canonical = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
probe_started_monotonic_ns = time.monotonic_ns()
probe_started_wall_time_ns = time.time_ns()
with open(sys.argv[2], "rb") as input_handle:
    input_bytes = input_handle.read()
schedule_path = None if sys.argv[3] == "-" else sys.argv[3]
schedule_artifact_sha256 = None
schedule_stat = None
if schedule_path is not None:
    with open(schedule_path, "rb") as schedule_handle:
        schedule_artifact_sha256 = hashlib.sha256(schedule_handle.read()).hexdigest()
    observed_stat = os.stat(schedule_path, follow_symlinks=True)
    schedule_stat = {
        "device": observed_stat.st_dev,
        "inode": observed_stat.st_ino,
        "size": observed_stat.st_size,
        "mtime_ns": observed_stat.st_mtime_ns,
        "ctime_ns": observed_stat.st_ctime_ns,
    }
cpu_model = "unknown"
try:
    with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
        for line in cpuinfo:
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
except OSError:
    cpu_model = platform.processor() or "unknown"
resource_environment = {key: environment.get(key) for key in resource_keys}
excluded_environment_keys = {
    "PWD",
    "UV_PROJECT",
    "FLASHPATCH_L7_SCHEDULE_SHA256", "FLASHPATCH_L7_SCHEDULE_SLOT",
    "FLASHPATCH_L7_SCHEDULE_ROUND", "FLASHPATCH_L7_SCHEDULE_POSITION",
    "FLASHPATCH_L7_SCHEDULE_COMPARATOR", "FLASHPATCH_L7_SCHEDULE_REPEAT",
}
effective_process_environment = {
    key: value for key, value in environment.items() if key not in excluded_environment_keys
}
visible_device_nodes = []
for device_root, directory_names, file_names in os.walk("/dev", followlinks=False):
    for file_name in file_names:
        device_path = os.path.join(device_root, file_name)
        try:
            mode = os.stat(device_path, follow_symlinks=False).st_mode
        except OSError:
            continue
        if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
            visible_device_nodes.append(device_path)
visible_device_nodes.sort()
effective_environment = {
    "process_environment": effective_process_environment,
    "machine": {
        "id": socket.gethostname(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
    },
    "cpu": {
        "model": cpu_model,
        "logical_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
    },
    "gpu": {
        "visible_device_nodes": visible_device_nodes,
    },
    "cache": {
        "policy": "WARM_INPUT_PRETOUCHED",
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_bytes": len(input_bytes),
    },
}
effective_canonical = json.dumps(
    effective_environment, sort_keys=True, separators=(",", ":")
).encode("utf-8")
payload = {
    "schema": "flashpatch-l7-child-runtime-probe-v1",
    "machine": effective_environment["machine"],
    "cpu_affinity": effective_environment["cpu"]["affinity"],
    "resource_environment": resource_environment,
    "effective_environment_policy_sha256": hashlib.sha256(
        json.dumps(
            {
                "schema": "flashpatch-l7-effective-environment-policy-v1",
                "compared_fields": [
                    "process_environment", "machine", "cpu.model",
                    "cpu.logical_count", "cpu.affinity", "gpu.visible_device_nodes", "cache.policy",
                    "cache.input_sha256", "cache.input_bytes",
                ],
                "excluded_environment_keys": [
                    "PWD", "UV_PROJECT", "FLASHPATCH_L7_SCHEDULE_SHA256",
                    "FLASHPATCH_L7_SCHEDULE_SLOT", "FLASHPATCH_L7_SCHEDULE_ROUND",
                    "FLASHPATCH_L7_SCHEDULE_POSITION", "FLASHPATCH_L7_SCHEDULE_COMPARATOR",
                    "FLASHPATCH_L7_SCHEDULE_REPEAT",
                ],
                "full_environment_treatment": "COMPARE_AFTER_DECLARED_IDENTITY_AND_SCHEDULE_EXCLUSIONS",
                "reason": "all non-identity process environment fields can affect resources or runtime behavior",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest(),
    "effective_environment": effective_environment,
    "effective_environment_sha256": hashlib.sha256(effective_canonical).hexdigest(),
    "full_environment_sha256": hashlib.sha256(canonical).hexdigest(),
    "launcher_identity_environment": {
        "PWD": environment.get("PWD"),
        "UV_PROJECT": environment.get("UV_PROJECT"),
    },
    "schedule_observation": {
        "path": schedule_path,
        "artifact_sha256": schedule_artifact_sha256,
        "stat": schedule_stat,
        "schedule_sha256": environment.get("FLASHPATCH_L7_SCHEDULE_SHA256"),
        "slot": environment.get("FLASHPATCH_L7_SCHEDULE_SLOT"),
        "round": environment.get("FLASHPATCH_L7_SCHEDULE_ROUND"),
        "position": environment.get("FLASHPATCH_L7_SCHEDULE_POSITION"),
        "comparator": environment.get("FLASHPATCH_L7_SCHEDULE_COMPARATOR"),
        "repeat_ordinal": environment.get("FLASHPATCH_L7_SCHEDULE_REPEAT"),
    } if schedule_path is not None else None,
}
tool_started_monotonic_ns = time.monotonic_ns()
completed = subprocess.run(sys.argv[4:], env=environment, check=False)
tool_finished_monotonic_ns = time.monotonic_ns()
payload["child_timing"] = {
    "probe_started_monotonic_ns": probe_started_monotonic_ns,
    "probe_started_wall_time_ns": probe_started_wall_time_ns,
    "tool_started_monotonic_ns": tool_started_monotonic_ns,
    "tool_finished_monotonic_ns": tool_finished_monotonic_ns,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
raise SystemExit(completed.returncode)
""".strip()
_FLASHPATCH_WORKER_SCRIPT = (
    "import sys; "
    "from flashpatch.external_league import _flashpatch_fair_worker; "
    "raise SystemExit(_flashpatch_fair_worker(*sys.argv[1:]))"
)

_TOOFLASHY_PARITY_ADAPTER_SCRIPT = r'''
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from tooflashy import analysis as upstream_analysis
from tooflashy import video as upstream_video


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def callable_evidence(value: object) -> dict[str, object]:
    source = inspect.getsource(value).encode("utf-8")
    module = sys.modules[value.__module__]
    module_path = Path(module.__file__).resolve()
    return {
        "module": value.__module__,
        "qualname": value.__qualname__,
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "callable_source_sha256": sha256_bytes(source),
    }


def executable_evidence(name: str) -> dict[str, object]:
    value = shutil.which(name)
    if value is None:
        raise SystemExit(f"required decoder executable is missing: {name}")
    path = Path(value).resolve()
    version = subprocess.run([str(path), "-version"], capture_output=True, check=False)
    if version.returncode != 0:
        raise SystemExit(f"decoder executable version probe failed: {name}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "version_stdout_sha256": sha256_bytes(version.stdout),
        "version_stderr_sha256": sha256_bytes(version.stderr),
    }


def distribution_evidence(name: str) -> dict[str, object] | None:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    rows = []
    distribution_root = Path(distribution.locate_file("")).resolve()
    for relative in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(relative)).resolve()
        try:
            path.relative_to(distribution_root)
        except ValueError:
            continue
        if not path.is_file() or path.name in {"RECORD", "INSTALLER", "REQUESTED"}:
            continue
        rows.append({
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "version": distribution.version,
        "file_count": len(rows),
        "files_sha256": sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: adapter VIDEO OUTPUT ADAPTER_SOURCE_SHA256")
    video_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    adapter_source_sha256 = sys.argv[3]
    if output_path.exists():
        raise SystemExit("adapter output already exists")
    iter_api = upstream_video.iter_video_frames
    analyze_api = upstream_analysis.analyze_frames
    if (
        iter_api.__module__ != "tooflashy.video"
        or iter_api.__qualname__ != "iter_video_frames"
        or analyze_api.__module__ != "tooflashy.analysis"
        or analyze_api.__qualname__ != "analyze_frames"
    ):
        raise SystemExit("upstream public API identity changed")

    fps, frames = iter_api(video_path, engine="ffmpeg")
    ledger: list[dict[str, object]] = []

    def audited_frames():
        for index, rgb in enumerate(frames):
            if (
                not isinstance(rgb, np.ndarray)
                or rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[-1] != 3
            ):
                raise TypeError("TooFlashy yielded a non-uint8 RGB ndarray")
            contiguous = np.ascontiguousarray(rgb)
            # This append is deliberately the final operation before yield.
            # The frozen script hash makes a post-consumption ledger a different adapter.
            ledger.append({
                "index": index,
                "cfr_timestamp_us": round(index * 1_000_000 / float(fps)),
                "shape": list(contiguous.shape),
                "pixel_format": "rgb24",
                "rgb_sha256": sha256_bytes(contiguous.tobytes()),
            })
            yield rgb

    result = analyze_api(audited_frames(), fps=fps, path=video_path)
    failures = list(result.failures)
    result_payload = {
        "path": str(result.path),
        "passes": result.passes,
        "fps": result.fps,
        "frame_count": result.frame_count,
        "event_count": len(result.events),
        "failures": failures,
        "event_representation": {
            "event_count": len(result.events),
            "failures": failures,
        },
    }
    dependency_versions = {}
    for distribution in ("tooflashy", "numpy", "opencv-python", "opencv-python-headless"):
        try:
            dependency_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    payload = {
        "schema": "flashpatch-l7-tooflashy-child-adapter-v1",
        "evidence_origin": "live_generator_append_immediately_before_yield_v1",
        "adapter_source_sha256": adapter_source_sha256,
        "input": {"path": str(video_path), "sha256": sha256_file(video_path)},
        "public_api": {
            "iter_video_frames": callable_evidence(iter_api),
            "analyze_frames": callable_evidence(analyze_api),
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "environment": dict(sorted(os.environ.items())),
            "sys_path": list(sys.path),
            "dependency_versions": dependency_versions,
            "dependency_evidence": {
                name: distribution_evidence(name)
                for name in ("tooflashy", "numpy", "opencv-python", "opencv-python-headless")
            },
            "decoder_executables": {
                "ffmpeg": executable_evidence("ffmpeg"),
                "ffprobe": executable_evidence("ffprobe"),
            },
        },
        "decode": {
            "engine": "ffmpeg",
            "fps": fps,
            "frame_count": len(ledger),
            "pixel_format": "rgb24",
            "ledger": ledger,
            "ledger_sha256": sha256_bytes(
                json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
        },
        "result": result_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


raise SystemExit(main())
'''.strip()

_KAYA_CONFORMANCE_CHILD_SCRIPT = r'''
from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import json
import math
import os
import stat
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REVISION = "0776ea3e6949a62d5becb8027a2765961b515793"
TREE = "34bef92fcc2bbd7c2e779475b84094493ae23aa1"
SOURCE_HASHES = {
    "LICENSE": "cd18b47f83e5cb4640272cc95e0144206be9ed55d8458573eba3cf0b49534da8",
    "requirements.txt": "da947acb06c423a8fb16487983383e98b19399f9fb585fdbe53e0c5bdc419961",
    "custom_video.py": "2704535a5c98d2f266ec71463a1d485dcaed088798ffc2fcbb2b9bc62e53630c",
    "PhotosensitivitySafetyEngine/engine/analysis.py": "397f3a3289118474708369dc338f4bc0f39e776d46af72cab8f3269d5d46cd79",
    "PhotosensitivitySafetyEngine/guidelines/w3c.py": "6b6053182bf683cca65e880c909f47f329378e462e9acb18667e659b9a0766ab",
    "PhotosensitivitySafetyEngine/libraries/common_functions.py": "b4991fbfe26fb2de44f255a3f17119123b310f710e215111857538e871678728",
    "PhotosensitivitySafetyEngine/libraries/custom_functions.py": "90fea09133824f63f7b419f187716dd906f20cd921c06e3ad78f43ba70519d77",
    "PhotosensitivitySafetyEngine/libraries/function_objects.py": "cba23e6f74bdb34de95a20a8ec09d42bcaf7ed65df2724f060b8dd7ad6e15e17",
}
REQUIRED_DISTRIBUTIONS = {
    "cycler": "0.12.1",
    "kiwisolver": "1.5.0",
    "matplotlib": "3.5.1",
    "numpy": "1.21.2",
    "opencv-python": "4.6.0.66",
    "packaging": "26.2",
    "pillow": "12.3.0",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
}
CALLABLE_SOURCE_HASHES = {
    "GuidelineProcess.analyse_file": "a6157a5aca4405ab35eb1a74c8d431b27a4c070f037ba2cebcc53017183a1830",
    "Display": "917eb81a7e931e22abbecdcd086c7555f17b910d212bd6530ca72733f4c7679e",
    "CustomVideo.frame_intervals": "9606b9a046de3a66659c4143f5343115a44baf3109ec8f32e482133d6ce57e02",
    "function_objects": "b52727ddc842a32eb2f13e2485daff3c4a8aff6801ae279353ffa6d311512ee1",
}
PIPELINE_SHA256 = "f5cb1f76491fa52c1500b18aa0edb8a5ed1ebd0b7b2f0ce4548f29f0c26ca391"
PYTHON_SHA256 = "685193c2432feb9d2a2b3ba129d976a7fd4172c60e1622b7b7ffd4308c40f6a5"
COMMON_BASE_RUNTIME_CLOSURE = {
    "classification": "PINNED_IMMUTABLE_RUNTIME_BASE_REQUIRES_STORAGE_INDEPENDENCE",
    "entry_count": 5422,
    "content_bytes": 80598705,
    "tree_sha256": "b2af88a9ea833e90a14d4769b1496c5d2c6332266231aa75b104afc75fd90608",
}
IMPORT_HOOKS = {
    "meta_path": [
        ["_frozen_importlib", "BuiltinImporter"],
        ["_frozen_importlib", "FrozenImporter"],
        ["_frozen_importlib_external", "PathFinder"],
        ["six", "_SixMetaPathImporter"],
    ],
    "path_hooks": [
        ["zipimport", "zipimporter"],
        ["_frozen_importlib_external", "FileFinder.path_hook.<locals>.path_hook_for_FileFinder"],
    ],
}
DISTRIBUTION_CLOSURES = {
    "cycler": {
        "normalized_record_sha256": "d336a450b93e33780693fa1ea208cbde9730d7cbfd98126e7743f2c5fa9af7c3",
        "portable_file_count": 8,
        "portable_files_sha256": "fb211d31b312a16478679126390102b80da524df05568d2510a5e62b27ca8a4d",
        "module_path": "lib/python3.10/site-packages/cycler/__init__.py",
        "module_sha256": "d4975182fe59cf1a3e5b57afec1fcb5aabb2b163fa2d91fa08793f08eb486971",
    },
    "kiwisolver": {
        "normalized_record_sha256": "5de1e73f73d6d2bb76d55a184db678c4f3fb820c1f28f2e35f865fdb825305c3",
        "portable_file_count": 11,
        "portable_files_sha256": "9e81212651d69c023a14030440caa70bd94332bc84c957a10d677a9b32a63819",
        "module_path": "lib/python3.10/site-packages/kiwisolver/__init__.py",
        "module_sha256": "0d42d2b355c4b47ac2c8b84dfe0f89f0f8af557b6f4b128dd97cf98116ca9122",
    },
    "matplotlib": {
        "normalized_record_sha256": "78d5c8b0244202a657b54e8bc8c35a148c87f159ef9e90090842b25743669320",
        "portable_file_count": 529,
        "portable_files_sha256": "3170b6596c2110656127f321ab9cca821f205d817500b03bbb8c535f97e5884a",
        "module_path": "lib/python3.10/site-packages/matplotlib/__init__.py",
        "module_sha256": "a19a045ba96b038b19844118d2e4e8aa281e5141398658b21376d9e0e7abe300",
    },
    "numpy": {
        "normalized_record_sha256": "7b65d299b0991f60aa703fadeeeeb64f143c6685f14c3c95179ea921b61bb088",
        "portable_file_count": 712,
        "portable_files_sha256": "16537179a9db995d2ae0b5b18120287ed4907120f5df41c2dc4ba53fd11ccef4",
        "module_path": "lib/python3.10/site-packages/numpy/__init__.py",
        "module_sha256": "02e54b39a5825a5eb95c74639b22593a8af816212f9bb1fdcb78f17112a5d27f",
    },
    "opencv-python": {
        "normalized_record_sha256": "752f90038f16bcf832cbd3dca3c34a56dad78f52049639427e09e4e275340861",
        "portable_file_count": 84,
        "portable_files_sha256": "8c6fa5be88fdacb5b57584b65903694fcf4745275386feed851ce9363818a15e",
        "module_path": "lib/python3.10/site-packages/cv2/__init__.py",
        "module_sha256": "3cc4a7613da6793421bb59e01d77148b03d37a774e44579c03d1c37407f0ebbe",
    },
    "packaging": {
        "normalized_record_sha256": "0291ff961cf9e2732b13da0c8cd8b2aa94d4935dd5275e5fce26d59c7de3e58e",
        "portable_file_count": 28,
        "portable_files_sha256": "a7a803628b4a59de7bc7d16cad012f7f131934566def906c5b4f092491cc6b2c",
        "module_path": "lib/python3.10/site-packages/packaging/__init__.py",
        "module_sha256": "42130474fbb65e882b2735774b42964bab7b97423d93c11e0d1265e1f9f0f3bb",
    },
    "pillow": {
        "normalized_record_sha256": "f0629a1d141a8be7b054b7c14a622d151b91ca3e8f212d4f61d3e6b78f69fb3d",
        "portable_file_count": 141,
        "portable_files_sha256": "e8ed90ad1888ea9010a5a21aafc7856c736c8c0c705461c698c1e1cd8747c9a8",
        "module_path": "lib/python3.10/site-packages/PIL/__init__.py",
        "module_sha256": "7361b6ad3878589affe5956e4a4da24d71398b0891fd3a8c15e9362217b4ca01",
    },
    "pyparsing": {
        "normalized_record_sha256": "3a25bda3ddedd53a3a3f97a924fb19753ada3145c233d640e564c7991e154dfc",
        "portable_file_count": 24,
        "portable_files_sha256": "6065751d0d53a20bf60b4d6d7273aceb2e5f0083b784f7d4aac5a4b2b9b2cd41",
        "module_path": "lib/python3.10/site-packages/pyparsing/__init__.py",
        "module_sha256": "08c1cce44cc9489825cfe3c2012bff9395b5e2d46486b09fe86488e0aacf2f6e",
    },
    "python-dateutil": {
        "normalized_record_sha256": "3f74cb322a1be616f4302ba3d7bcdd4f73119545a8905b74d939ec35f810977c",
        "portable_file_count": 26,
        "portable_files_sha256": "19c57a119691ea1520c41786da4e2bf3838364a6d84627c882826cffb23bc68c",
        "module_path": "lib/python3.10/site-packages/dateutil/__init__.py",
        "module_sha256": "32a6a6ebb58ef4891399417223aeaf4ba2284974e9f46dfcf0369d1f62c230b6",
    },
    "six": {
        "normalized_record_sha256": "8e98d2d8310169ea395e9085f3c471323f478ed9377ebad2a41f56331df15675",
        "portable_file_count": 7,
        "portable_files_sha256": "fc9794407ab44b916d36ea7022d0f5e22f12706d4255b61ab5b23b8850ca5885",
        "module_path": "lib/python3.10/site-packages/six.py",
        "module_sha256": "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def hook_identity(value: object) -> list[str]:
    if isinstance(value, type):
        return [value.__module__, value.__qualname__]
    return [value.__module__, getattr(value, "__qualname__", type(value).__qualname__)]


def runtime_base_evidence(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise SystemExit("Kaya interpreter base is unavailable")
    rows = []
    content_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise SystemExit("Kaya interpreter base contains an absolute symlink")
            try:
                path.resolve(strict=False).relative_to(root)
            except ValueError as exc:
                raise SystemExit("Kaya interpreter base symlink escapes its root") from exc
            rows.append({"path": relative, "type": "symlink", "target": target})
        elif path.is_file():
            size = path.stat().st_size
            content_bytes += size
            rows.append({
                "path": relative,
                "type": "file",
                "bytes": size,
                "sha256": sha256_file(path),
            })
        elif not path.is_dir():
            raise SystemExit("Kaya interpreter base contains a non-file artifact")
    evidence = {
        "classification": "PINNED_IMMUTABLE_RUNTIME_BASE_REQUIRES_STORAGE_INDEPENDENCE",
        "entry_count": len(rows),
        "content_bytes": content_bytes,
        "tree_sha256": canonical_sha256(rows),
    }
    if evidence != COMMON_BASE_RUNTIME_CLOSURE:
        raise SystemExit("Kaya interpreter base closure drifted")
    return evidence


def loaded_module_census(
    checkout: Path,
    environment_root: Path,
    interpreter_root: Path,
    site_packages: Path,
) -> dict[str, object]:
    rows = []
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit("Kaya loaded module path is unavailable: " + module_name)
        roots = (
            ("upstream_source", checkout),
            ("site_packages", site_packages),
            ("interpreter_base", interpreter_root),
        )
        binding = None
        for classification, root in roots:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            binding = (classification, relative)
            break
        if binding is None:
            raise SystemExit("Kaya loaded module escapes frozen source and runtime: " + module_name)
        classification, relative = binding
        rows.append({
            "module": module_name,
            "classification": classification,
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "shared_object": path.suffix == ".so",
        })
    hooks = {
        "meta_path": [hook_identity(value) for value in sys.meta_path],
        "path_hooks": [hook_identity(value) for value in sys.path_hooks],
    }
    if hooks != IMPORT_HOOKS:
        raise SystemExit("Kaya import-hook identity drifted")
    return {
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "module_count": len(rows),
        "shared_object_count": sum(bool(row["shared_object"]) for row in rows),
        "import_hooks": hooks,
    }


def imported_distribution_census(
    environment_root: Path,
    site_packages: Path,
) -> dict[str, object]:
    owners: dict[Path, set[str]] = {}
    for distribution_name in REQUIRED_DISTRIBUTIONS:
        distribution = metadata.distribution(distribution_name)
        for relative in distribution.files or []:
            path = Path(distribution.locate_file(relative)).resolve()
            owners.setdefault(path, set()).add(distribution_name)
    rows = []
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).resolve()
        try:
            path.relative_to(site_packages)
        except ValueError:
            continue
        path_owners = sorted(owners.get(path, set()))
        if len(path_owners) != 1:
            raise SystemExit("Kaya imported site-packages module lacks one frozen RECORD owner: " + module_name)
        rows.append({
            "module": module_name,
            "path": str(path.relative_to(environment_root)),
            "distribution": path_owners[0],
            "sha256": sha256_file(path),
        })
    imported_distributions = sorted({row["distribution"] for row in rows})
    if imported_distributions != sorted(REQUIRED_DISTRIBUTIONS):
        raise SystemExit("Kaya actual imported distribution census differs from the frozen closure")
    return {
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "distributions": imported_distributions,
        "module_count": len(rows),
        "all_site_modules_record_owned": True,
    }


def source_audit(checkout: Path) -> dict[str, object]:
    if not checkout.is_dir():
        raise SystemExit("Kaya checkout is missing")
    for relative, expected in SOURCE_HASHES.items():
        path = (checkout / relative).resolve()
        try:
            path.relative_to(checkout)
        except ValueError:
            raise SystemExit("Kaya source path escapes checkout")
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit("Kaya source hash mismatch: " + relative)
    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=checkout,
        capture_output=True, text=True, check=False,
    )
    tree = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD^{tree}"], cwd=checkout,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=checkout,
        capture_output=True, text=True, check=False,
    )
    if (
        head.returncode != 0 or head.stdout.strip() != REVISION
        or tree.returncode != 0 or tree.stdout.strip() != TREE
        or status.returncode != 0 or status.stdout != ""
    ):
        raise SystemExit("Kaya checkout revision, tree, or clean-state mismatch")
    return {
        "revision": REVISION,
        "tree": TREE,
        "source_hashes": SOURCE_HASHES,
        "license": "BSD-3-Clause",
    }


def distribution_evidence(name: str) -> dict[str, object]:
    distribution = metadata.distribution(name)
    if distribution.version != REQUIRED_DISTRIBUTIONS[name]:
        raise SystemExit("Kaya dependency version mismatch: " + name)
    environment_root = Path(sys.executable).parent.parent.resolve()
    files = list(distribution.files or [])
    record_entries = [relative for relative in files if Path(str(relative)).name == "RECORD"]
    if len(record_entries) != 1:
        raise SystemExit("Kaya dependency RECORD identity is ambiguous: " + name)
    record_path = Path(distribution.locate_file(record_entries[0])).resolve()
    try:
        record_path.relative_to(environment_root)
    except ValueError as exc:
        raise SystemExit("Kaya dependency RECORD escapes isolated environment: " + name) from exc
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            record_rows = list(csv.reader(handle))
    except (OSError, csv.Error) as exc:
        raise SystemExit("Kaya dependency RECORD is unreadable: " + name) from exc
    if not record_rows or any(len(row) != 3 for row in record_rows):
        raise SystemExit("Kaya dependency RECORD row shape is invalid: " + name)
    record_names = [row[0] for row in record_rows]
    if len(record_names) != len(set(record_names)) or set(record_names) != {str(value) for value in files}:
        raise SystemExit("Kaya dependency RECORD has missing, duplicate, or extra entries: " + name)
    installed_rows = []
    portable_rows = []
    normalized_record_rows = []
    installed_paths = set()
    owned_roots = set()
    site_packages = record_path.parent.parent.resolve()
    for relative, declared_hash, declared_size in record_rows:
        installed = (site_packages / relative).resolve()
        try:
            installed.relative_to(environment_root)
        except ValueError as exc:
            raise SystemExit("Kaya dependency file escapes isolated environment: " + name) from exc
        if not installed.is_file():
            raise SystemExit("Kaya dependency RECORD file is missing: " + name + ":" + relative)
        data = installed.read_bytes()
        if declared_size and (not declared_size.isdigit() or int(declared_size) != len(data)):
            raise SystemExit("Kaya dependency RECORD size mismatches: " + name + ":" + relative)
        if declared_hash:
            try:
                algorithm, encoded = declared_hash.split("=", 1)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
            except ValueError as exc:
                raise SystemExit("Kaya dependency RECORD digest is invalid: " + name) from exc
            if algorithm != "sha256" or encoded != expected:
                raise SystemExit("Kaya dependency RECORD digest mismatches: " + name + ":" + relative)
        installed_paths.add(installed)
        installed_rows.append({
            "path": str(installed.relative_to(environment_root)),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        })
        if site_packages in installed.parents and installed != record_path:
            portable_rows.append({
                "path": str(installed.relative_to(site_packages)),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            })
        if site_packages not in installed.parents:
            normalized_record_rows.append([relative, "<relocatable-generated>", "<relocatable-generated>"])
        elif installed == record_path:
            normalized_record_rows.append([relative, "", ""])
        else:
            normalized_record_rows.append([relative, declared_hash, declared_size])
        if installed == record_path or site_packages in installed.parents:
            relative_to_site = installed.relative_to(site_packages)
            if relative_to_site.parts:
                owned_roots.add((site_packages / relative_to_site.parts[0]).resolve())
    for owned_root in owned_roots:
        candidates = [owned_root] if owned_root.is_file() else list(owned_root.rglob("*"))
        for candidate in candidates:
            if (
                candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix != ".pyc"
                and candidate.resolve() not in installed_paths
            ):
                raise SystemExit("Kaya dependency installed tree has an unrecorded extra file: " + name)
    installed_rows.sort(key=lambda row: row["path"])
    portable_rows.sort(key=lambda row: row["path"])
    normalized_record_rows.sort()
    return {
        "version": distribution.version,
        "record": {
            "path": str(record_path.relative_to(environment_root)),
            "sha256": sha256_file(record_path),
            "entry_count": len(record_rows),
        },
        "owned_roots": sorted(str(value.relative_to(environment_root)) for value in owned_roots),
        "file_count": len(installed_rows),
        "files_sha256": canonical_sha256(installed_rows),
        "portable_file_count": len(portable_rows),
        "portable_files_sha256": canonical_sha256(portable_rows),
        "normalized_record_sha256": canonical_sha256(normalized_record_rows),
        "record_hashes_verified": True,
        "unrecorded_files_absent": True,
    }


def callable_evidence(value: object) -> dict[str, object]:
    module_path = Path(sys.modules[value.__module__].__file__).resolve()
    return {
        "module": value.__module__,
        "qualname": value.__qualname__,
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "callable_source_sha256": sha256_bytes(inspect.getsource(value).encode("utf-8")),
    }


def load_direct_input(path: Path) -> tuple[object, dict[str, object]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("Kaya direct manifest is unreadable") from exc
    expected_fields = {
        "schema", "fixture_id", "fps", "frame_count", "shape", "dtype",
        "pixel_format", "frames", "frames_file_sha256", "raw_rgb_sha256",
        "ledger", "ledger_sha256", "canonical_video", "conversion_receipt",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise SystemExit("Kaya direct manifest fields are invalid")
    if (
        manifest.get("schema") != "flashpatch-l7-kaya-direct-rgb-input-v1"
        or isinstance(manifest.get("fps"), bool)
        or not isinstance(manifest.get("fps"), int)
        or manifest.get("fps") != 60
        or manifest.get("dtype") != "uint8"
        or manifest.get("pixel_format") != "rgb24"
    ):
        raise SystemExit("Kaya direct manifest is not exact 60 CFR uint8 RGB")
    frames_path = Path(str(manifest.get("frames", ""))).resolve()
    if not frames_path.is_file() or manifest.get("frames_file_sha256") != sha256_file(frames_path):
        raise SystemExit("Kaya direct RGB frame artifact hash mismatch")
    import numpy as np
    try:
        frames = np.load(frames_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit("Kaya direct RGB frame artifact is unreadable") from exc
    frame_count = manifest.get("frame_count")
    if (
        not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
        or frames.ndim != 4 or frames.shape[-1] != 3
        or isinstance(frame_count, bool) or not isinstance(frame_count, int)
        or frame_count <= 0 or len(frames) != frame_count
        or list(frames.shape) != manifest.get("shape")
        or sha256_bytes(np.ascontiguousarray(frames).tobytes()) != manifest.get("raw_rgb_sha256")
    ):
        raise SystemExit("Kaya direct RGB array contract is invalid")
    ledger = manifest.get("ledger")
    if not isinstance(ledger, list) or len(ledger) != frame_count or canonical_sha256(ledger) != manifest.get("ledger_sha256"):
        raise SystemExit("Kaya direct RGB ledger is invalid")
    seen = set()
    for index, row in enumerate(ledger):
        expected = {
            "index": index,
            "cfr_timestamp_us": round(index * 1_000_000 / 60),
            "shape": list(frames[index].shape),
            "pixel_format": "rgb24",
            "rgb_sha256": sha256_bytes(np.ascontiguousarray(frames[index]).tobytes()),
        }
        if not isinstance(row, dict) or row != expected or row["index"] in seen:
            raise SystemExit("Kaya direct RGB ledger row is missing, duplicate, or malformed")
        seen.add(row["index"])
    return frames, manifest


def normalize_result(values: object, custom_video_class: object) -> dict[str, object]:
    if not isinstance(values, dict) or set(values) != {"General Flashes", "Red Flashes"}:
        raise SystemExit("Kaya upstream result fields changed")
    general = [float(value) for value in values["General Flashes"]]
    red = [float(value) for value in values["Red Flashes"]]
    if len(general) != len(red) or not all(math.isfinite(value) for value in general + red):
        raise SystemExit("Kaya upstream raw arrays are invalid")
    # Call the upstream interval implementation verbatim.  In particular, its
    # EOF-open `both` omission is retained instead of repaired by the adapter.
    interval_owner = custom_video_class.__new__(custom_video_class)
    intervals = custom_video_class.frame_intervals(interval_owner, values)
    if intervals is None:
        intervals = []
    return {
        "raw": {"General Flashes": general, "Red Flashes": red},
        "interval_tuples": [list(value) for value in intervals],
        "interval_semantics": "unmodified_CustomVideo.frame_intervals_including_eof_open_both_bug",
    }


def main() -> int:
    if len(sys.argv) != 8:
        raise SystemExit("usage: child MODE CHECKOUT VIDEO DIRECT_MANIFEST OUTPUT ADAPTER_SHA REUSE_COUNT")
    mode = sys.argv[1]
    checkout = Path(sys.argv[2]).resolve()
    video = Path(sys.argv[3]).resolve()
    direct_manifest_path = None if sys.argv[4] == "-" else Path(sys.argv[4]).resolve()
    output = Path(sys.argv[5]).resolve()
    adapter_sha256 = sys.argv[6]
    try:
        reuse_count = int(sys.argv[7])
    except ValueError as exc:
        raise SystemExit("Kaya reuse count is invalid") from exc
    if mode not in {"native", "direct"} or output.exists() or reuse_count not in {1, 2}:
        raise SystemExit("Kaya child mode, output, or reuse count is invalid")
    if mode == "native" and (direct_manifest_path is not None or reuse_count != 1):
        raise SystemExit("Kaya native path cannot receive direct input or reuse")
    if mode == "direct" and direct_manifest_path is None:
        raise SystemExit("Kaya direct path requires a frozen RGB manifest")
    if sys.version_info.major != 3 or sys.version_info.minor > 10 or sys.version_info.minor < 8:
        raise SystemExit("Kaya upstream requires Python 3.8 through 3.10")
    if sys.flags.no_site != 1 or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise SystemExit("Kaya child must run with -S and no site customization")
    provenance = source_audit(checkout)
    environment_root = Path(sys.executable).parent.parent.resolve()
    interpreter_root = Path(sys.base_prefix).resolve()
    site_packages = environment_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if not site_packages.is_dir():
        raise SystemExit("Kaya isolated site-packages directory is missing")
    standard_library_paths = []
    for item in sys.path:
        if not item:
            continue
        resolved = Path(item).resolve()
        if resolved == interpreter_root or interpreter_root in resolved.parents:
            standard_library_paths.append(str(resolved))
    sys.path[:] = [str(checkout), *standard_library_paths, str(site_packages)]
    import cv2
    import matplotlib
    import numpy as np
    from PhotosensitivitySafetyEngine.engine import analysis as analysis_module
    from PhotosensitivitySafetyEngine.engine.analysis import Display, GuidelineProcess
    from PhotosensitivitySafetyEngine.guidelines import w3c
    from custom_video import CustomVideo

    if (
        np.__version__ != "1.21.2" or cv2.__version__ != "4.6.0"
        or matplotlib.__version__ != "3.5.1"
        or type(w3c.w3c_guideline) is not GuidelineProcess
        or not inspect.ismethod(w3c.w3c_guideline.analyse_file)
        or w3c.w3c_guideline.analyse_file.__func__ is not GuidelineProcess.analyse_file
        or w3c.w3c_guideline.objects is not w3c.function_objects
        or w3c.w3c_guideline.pipeline is not w3c.processing_pipeline
    ):
        raise SystemExit("Kaya imported dependency or guideline identity mismatch")
    dependencies = {name: distribution_evidence(name) for name in REQUIRED_DISTRIBUTIONS}
    imported_modules = {
        "cycler": Path(sys.modules["cycler"].__file__).resolve(),
        "kiwisolver": Path(sys.modules["kiwisolver"].__file__).resolve(),
        "matplotlib": Path(matplotlib.__file__).resolve(),
        "numpy": Path(np.__file__).resolve(),
        "opencv-python": Path(cv2.__file__).resolve(),
        "packaging": Path(sys.modules["packaging"].__file__).resolve(),
        "pillow": Path(sys.modules["PIL"].__file__).resolve(),
        "pyparsing": Path(sys.modules["pyparsing"].__file__).resolve(),
        "python-dateutil": Path(sys.modules["dateutil"].__file__).resolve(),
        "six": Path(sys.modules["six"].__file__).resolve(),
    }
    if sha256_file(Path(sys.executable).resolve()) != PYTHON_SHA256:
        raise SystemExit("Kaya Python executable differs from the frozen environment")
    for distribution_name, module_path in imported_modules.items():
        expected = DISTRIBUTION_CLOSURES[distribution_name]
        evidence = dependencies[distribution_name]
        if (
            evidence["normalized_record_sha256"] != expected["normalized_record_sha256"]
            or evidence["portable_file_count"] != expected["portable_file_count"]
            or evidence["portable_files_sha256"] != expected["portable_files_sha256"]
            or str(module_path.relative_to(environment_root)) != expected["module_path"]
            or sha256_file(module_path) != expected["module_sha256"]
        ):
            raise SystemExit("Kaya immutable dependency closure drifted: " + distribution_name)
        distribution = metadata.distribution(distribution_name)
        recorded_paths = {
            Path(distribution.locate_file(relative)).resolve()
            for relative in distribution.files or []
        }
        if module_path not in recorded_paths:
            raise SystemExit("Kaya imported module is not owned by its frozen RECORD: " + distribution_name)
    default_display = Display()
    display_contract = {
        "display_resolution": list(default_display.get_property("display_resolution")),
        "display_diameter": default_display.get_property("display_diameter"),
        "display_distance": default_display.get_property("display_distance"),
        "frame_rate_before_input": default_display.get_property("frame_rate"),
        "candelas": default_display.get_property("candelas"),
        "speedup": 10,
        "expected_analysis_resolution": [102, 76],
    }
    if display_contract != {
        "display_resolution": [1024, 768], "display_diameter": 16,
        "display_distance": 24, "frame_rate_before_input": 30,
        "candelas": 200, "speedup": 10,
        "expected_analysis_resolution": [102, 76],
    }:
        raise SystemExit("Kaya upstream Display defaults changed")
    pipeline_projection = [
        [row[0], list(row[1]) if isinstance(row[1], tuple) else row[1], *list(row[2:])]
        for row in w3c.processing_pipeline
    ]
    callable_bindings = {
        "GuidelineProcess.analyse_file": callable_evidence(GuidelineProcess.analyse_file),
        "Display": callable_evidence(Display),
        "CustomVideo.frame_intervals": callable_evidence(CustomVideo.frame_intervals),
        "function_objects": callable_evidence(w3c.function_objects),
    }
    if (
        [row[0] for row in pipeline_projection[-2:]] != ["fullFlashCountGeneral", "fullFlashCountRed"]
        or canonical_sha256(pipeline_projection) != PIPELINE_SHA256
        or any(
            callable_bindings[label]["callable_source_sha256"] != expected
            for label, expected in CALLABLE_SOURCE_HASHES.items()
        )
    ):
        raise SystemExit("Kaya W3C callable or pipeline source identity changed")

    direct_frames = None
    direct_manifest = None
    if direct_manifest_path is not None:
        direct_frames, direct_manifest = load_direct_input(direct_manifest_path)
    original_capture = cv2.VideoCapture
    capture_runs: list[dict[str, object]] = []

    class NativeCapture:
        def __init__(self, path: object):
            self.inner = original_capture(path)
            self.ledger: list[dict[str, object]] = []
            self.observed_fps = float(self.inner.get(cv2.CAP_PROP_FPS))
            self.observed_frame_count = int(self.inner.get(cv2.CAP_PROP_FRAME_COUNT))
            capture_runs.append({
                "ledger": self.ledger,
                "observed_fps": self.observed_fps,
                "observed_frame_count": self.observed_frame_count,
            })

        def read(self):
            check, frame = self.inner.read()
            if not check:
                return check, frame
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise TypeError("Kaya native decoder yielded non-uint8 BGR")
            contiguous = np.ascontiguousarray(frame)
            self.ledger.append({
                "index": len(self.ledger), "shape": list(contiguous.shape),
                "pixel_format": "bgr24", "bgr_sha256": sha256_bytes(contiguous.tobytes()),
            })
            return True, frame

        def get(self, prop):
            return self.inner.get(prop)

        def release(self):
            return self.inner.release()

    class DirectCapture:
        def __init__(self, path: object):
            if direct_frames is None or direct_manifest is None:
                raise RuntimeError("Kaya direct capture lacks RGB input")
            # This is the adapter's sole channel conversion.  The frozen
            # script and per-frame RGB/BGR hash pairs make a second reversal a
            # different, detectable adapter.
            self.frames = np.ascontiguousarray(direct_frames[..., ::-1])
            self.cursor = 0
            self.ledger: list[dict[str, object]] = []
            capture_runs.append({
                "ledger": self.ledger,
                "observed_fps": 60.0,
                "observed_frame_count": len(self.frames),
            })

        def read(self):
            if self.cursor >= len(self.frames):
                return False, None
            frame = self.frames[self.cursor]
            contiguous = np.ascontiguousarray(frame)
            self.ledger.append({
                "index": self.cursor, "shape": list(contiguous.shape),
                "pixel_format": "bgr24", "bgr_sha256": sha256_bytes(contiguous.tobytes()),
            })
            self.cursor += 1
            return True, frame.copy()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 60.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return float(len(self.frames))
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return float(self.cursor)
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.frames.shape[2])
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.frames.shape[1])
            return 0.0

        def release(self):
            return None

    analysis_module.cv2.VideoCapture = NativeCapture if mode == "native" else DirectCapture
    results = []
    try:
        for _ in range(reuse_count):
            values = w3c.w3c_guideline.analyse_file(
                str(video) if mode == "native" else "DIRECT_RGB_LEDGER",
                show_live_chart=False, show_dsp=False, show_analysis=False,
            )
            results.append(normalize_result(values, CustomVideo))
    finally:
        analysis_module.cv2.VideoCapture = original_capture
    if len(capture_runs) != reuse_count:
        raise SystemExit("Kaya capture invocation count mismatch")
    for run_index, (capture, result) in enumerate(zip(capture_runs, results)):
        if (
            capture["observed_fps"] != 60.0
            or capture["observed_frame_count"] != len(capture["ledger"])
            or len(result["raw"]["General Flashes"]) != len(capture["ledger"])
            or len(result["raw"]["Red Flashes"]) != len(capture["ledger"])
        ):
            raise SystemExit("Kaya capture/result 60 CFR frame count mismatch")
        capture["ledger_sha256"] = canonical_sha256(capture["ledger"])

    direct_conversion = None
    if mode == "direct":
        rgb_ledger = direct_manifest["ledger"]
        bgr_ledger = capture_runs[0]["ledger"]
        direct_conversion = {
            "operation": "rgb_to_bgr_channel_reverse_once",
            "pairs": [
                {
                    "index": index,
                    "rgb_sha256": rgb_ledger[index]["rgb_sha256"],
                    "bgr_sha256": bgr_ledger[index]["bgr_sha256"],
                }
                for index in range(len(rgb_ledger))
            ],
        }
        direct_conversion["pairs_sha256"] = canonical_sha256(direct_conversion["pairs"])
    import_census = imported_distribution_census(environment_root, site_packages)
    runtime_base = runtime_base_evidence(interpreter_root)
    loaded_modules = loaded_module_census(
        checkout, environment_root, interpreter_root, site_packages,
    )
    payload = {
        "schema": "flashpatch-l7-kaya-conformance-child-v1",
        "identity": "KAYA_SOURCE_DIRECT_INPUT_PROTOTYPE_0776EA3E_UNSCORED",
        "classification": "UNSCORED_CONFORMANCE_ONLY",
        "adapter_source_sha256": adapter_sha256,
        "mode": mode,
        "reuse_count": reuse_count,
        "upstream": provenance,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "python_version_info": list(sys.version_info[:3]),
            "sys_prefix": str(Path(sys.prefix).resolve()),
            "base_prefix": str(interpreter_root),
            "environment_root": str(environment_root),
            "site_packages": str(site_packages),
            "no_site": bool(sys.flags.no_site),
            "sys_path": list(sys.path),
            "dependencies": dependencies,
            "imported_modules": {
                name: {"path": str(module_path), "sha256": sha256_file(module_path)}
                for name, module_path in sorted(imported_modules.items())
            },
            "import_census": import_census,
            "runtime_base": runtime_base,
            "loaded_modules": loaded_modules,
        },
        "api": {
            **callable_bindings,
            "guideline_object_module": type(w3c.w3c_guideline).__module__,
            "guideline_object_qualname": type(w3c.w3c_guideline).__qualname__,
            "function_objects_module": w3c.function_objects.__module__,
            "pipeline": pipeline_projection,
            "pipeline_sha256": canonical_sha256(pipeline_projection),
            "display_defaults": display_contract,
        },
        "input": {
            "video": str(video),
            "video_sha256": sha256_file(video),
            "direct_manifest": str(direct_manifest_path) if direct_manifest_path is not None else None,
            "direct_manifest_sha256": sha256_file(direct_manifest_path) if direct_manifest_path is not None else None,
        },
        "capture_runs": capture_runs,
        "results": results,
        "direct_conversion": direct_conversion,
        "claim_boundary": {
            "scoreable": False,
            "population_authorized": False,
            "participant_status": "UNSCORED_PROTOTYPE",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


raise SystemExit(main())
'''.strip()

_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP = r'''
#include <iris/Configuration.h>
#include <iris/FrameData.h>
#include <iris/Log.h>
#include <iris/VideoAnalyser.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openssl/sha.h>

#include <array>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef FLASHPATCH_ADAPTER_SOURCE_SHA256
#error FLASHPATCH_ADAPTER_SOURCE_SHA256 must bind the generated adapter source
#endif

namespace {

std::string hex_bytes(const unsigned char* bytes, std::size_t size) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < size; ++index) {
        stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return stream.str();
}

std::string sha256_hex(const unsigned char* bytes, std::size_t size) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    if (SHA256(bytes, size, digest.data()) == nullptr) {
        throw std::runtime_error("SHA256 failed");
    }
    return hex_bytes(digest.data(), digest.size());
}

std::string file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file for SHA256: " + path);
    }
    SHA256_CTX context;
    if (SHA256_Init(&context) != 1) {
        throw std::runtime_error("SHA256_Init failed");
    }
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 && SHA256_Update(&context, buffer.data(), static_cast<std::size_t>(count)) != 1) {
            throw std::runtime_error("SHA256_Update failed");
        }
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    if (SHA256_Final(digest.data(), &context) != 1) {
        throw std::runtime_error("SHA256_Final failed");
    }
    return hex_bytes(digest.data(), digest.size());
}

std::string mat_sha256(const cv::Mat& frame) {
    if (!frame.isContinuous()) {
        throw std::runtime_error("frame is not contiguous at hash boundary");
    }
    return sha256_hex(frame.ptr<unsigned char>(0), frame.total() * frame.elemSize());
}

const char* flash_result_name(iris::FlashResult value) {
    switch (value) {
        case iris::FlashResult::Pass: return "Pass";
        case iris::FlashResult::PassWithWarning: return "PassWithWarning";
        case iris::FlashResult::ExtendedFail: return "ExtendedFail";
        case iris::FlashResult::FlashFail: return "FlashFail";
    }
    throw std::runtime_error("unknown FlashResult value");
}

const char* pattern_result_name(iris::PatternResult value) {
    switch (value) {
        case iris::PatternResult::Pass: return "Pass";
        case iris::PatternResult::Fail: return "Fail";
    }
    throw std::runtime_error("unknown PatternResult value");
}

unsigned int parse_frame_count(const char* text) {
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(text, &consumed, 10);
    if (consumed != std::string(text).size() || parsed == 0 || parsed > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("expected frame count is invalid");
    }
    return static_cast<unsigned int>(parsed);
}

int parse_positive_int(const char* text, const char* label) {
    std::size_t consumed = 0;
    const long parsed = std::stol(text, &consumed, 10);
    if (consumed != std::string(text).size() || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(label) + " is invalid");
    }
    return static_cast<int>(parsed);
}

class ReadOnlyMappedFile {
public:
    explicit ReadOnlyMappedFile(const std::string& path) {
        m_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (m_fd < 0) {
            throw std::runtime_error("cannot open regular input without following links: " + path);
        }
        struct stat value {};
        if (::fstat(m_fd, &value) != 0 || !S_ISREG(value.st_mode) || value.st_size <= 0) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("input is not a non-empty regular file");
        }
        if (static_cast<std::uintmax_t>(value.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("input is too large for this address space");
        }
        m_size = static_cast<std::size_t>(value.st_size);
        m_device = static_cast<std::uint64_t>(value.st_dev);
        m_inode = static_cast<std::uint64_t>(value.st_ino);
        m_modified_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec);
        m_modified_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec);
        void* mapped = ::mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("cannot map read-only input");
        }
        m_data = static_cast<const unsigned char*>(mapped);
    }

    ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
    ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;

    ~ReadOnlyMappedFile() {
        if (m_data != nullptr) {
            ::munmap(const_cast<unsigned char*>(m_data), m_size);
        }
        if (m_fd >= 0) {
            ::close(m_fd);
        }
    }

    const unsigned char* data() const { return m_data; }
    std::size_t size() const { return m_size; }
    bool unchanged() const {
        struct stat value {};
        return ::fstat(m_fd, &value) == 0
            && S_ISREG(value.st_mode)
            && static_cast<std::uint64_t>(value.st_dev) == m_device
            && static_cast<std::uint64_t>(value.st_ino) == m_inode
            && static_cast<std::uint64_t>(value.st_size) == m_size
            && static_cast<std::int64_t>(value.st_mtim.tv_sec) == m_modified_seconds
            && static_cast<std::int64_t>(value.st_mtim.tv_nsec) == m_modified_nanoseconds;
    }

private:
    int m_fd = -1;
    const unsigned char* m_data = nullptr;
    std::size_t m_size = 0;
    std::uint64_t m_device = 0;
    std::uint64_t m_inode = 0;
    std::int64_t m_modified_seconds = 0;
    std::int64_t m_modified_nanoseconds = 0;
};

struct FileIdentity {
    std::uint64_t device;
    std::uint64_t inode;
    std::uint64_t size;
    std::int64_t modified_seconds;
    std::int64_t modified_nanoseconds;
};

FileIdentity file_identity(const std::string& path) {
    struct stat value {};
    if (::lstat(path.c_str(), &value) != 0 || !S_ISREG(value.st_mode)) {
        throw std::runtime_error("configuration path is not a regular file");
    }
    return {
        static_cast<std::uint64_t>(value.st_dev),
        static_cast<std::uint64_t>(value.st_ino),
        static_cast<std::uint64_t>(value.st_size),
        static_cast<std::int64_t>(value.st_mtim.tv_sec),
        static_cast<std::int64_t>(value.st_mtim.tv_nsec),
    };
}

bool operator==(const FileIdentity& left, const FileIdentity& right) {
    return left.device == right.device
        && left.inode == right.inode
        && left.size == right.size
        && left.modified_seconds == right.modified_seconds
        && left.modified_nanoseconds == right.modified_nanoseconds;
}

void require_exact_object_fields(
    const nlohmann::json& value,
    std::initializer_list<const char*> fields,
    const char* label
) {
    if (!value.is_object() || value.size() != fields.size()) {
        throw std::runtime_error(std::string(label) + " fields are invalid");
    }
    for (const char* field : fields) {
        if (!value.contains(field)) {
            throw std::runtime_error(std::string(label) + " omits field: " + field);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 12) {
        std::cerr << "usage: iris-source-adapter RAW_RGB TIMELINE CONFIG_DIR OUTPUT WIDTH HEIGHT EXPECTED_FRAMES EXPECTED_RAW_SHA256 EXPECTED_TIMELINE_SHA256 EXPECTED_CONFIG_SHA256 THREAD_LIMIT\n";
        return 2;
    }
    bool log_initialized = false;
    bool analyser_initialized = false;
    iris::Configuration configuration;
    std::unique_ptr<iris::VideoAnalyser> analyser;
    try {
        const std::string raw_rgb_path = argv[1];
        const std::string timeline_path = argv[2];
        std::string config_directory = argv[3];
        const std::string output_path = argv[4];
        const int width = parse_positive_int(argv[5], "RGB frame width");
        const int height = parse_positive_int(argv[6], "RGB frame height");
        const unsigned int expected_frames = parse_frame_count(argv[7]);
        const std::string expected_raw_sha256 = argv[8];
        const std::string expected_timeline_sha256 = argv[9];
        const std::string expected_config_sha256 = argv[10];
        const int thread_limit = parse_positive_int(argv[11], "OpenCV analysis thread limit");
        if (thread_limit > 256) {
            throw std::runtime_error("OpenCV analysis thread limit is invalid");
        }
        if (config_directory.empty() || config_directory.back() != '/') {
            throw std::runtime_error("configuration directory must end with slash");
        }
        const std::string config_path = config_directory + "appsettings.json";
        const auto config_identity_before = file_identity(config_path);
        if (file_sha256(config_path) != expected_config_sha256) {
            throw std::runtime_error("IRIS configuration hash drifted before initialization");
        }

        // Fail closed before initializing IRIS or calling AnalyseFrame.  The
        // entire raw slate and every timestamp/hash row are validated first.
        const ReadOnlyMappedFile raw_rgb(raw_rgb_path);
        const ReadOnlyMappedFile timeline_bytes(timeline_path);
        if (sha256_hex(timeline_bytes.data(), timeline_bytes.size()) != expected_timeline_sha256) {
            throw std::runtime_error("RGB timeline bytes drifted before parsing");
        }
        const auto timeline = nlohmann::json::parse(
            timeline_bytes.data(), timeline_bytes.data() + timeline_bytes.size(), nullptr, true, false
        );
        require_exact_object_fields(
            timeline,
            {"schema", "source_video", "conversion_receipt", "renderer_source", "decoder", "raw_rgb", "fps", "frame_count", "frames"},
            "RGB timeline ledger"
        );
        if (timeline.at("schema") != "flashpatch-l7-ea-iris-direct-rgb-input-v1") {
            throw std::runtime_error("RGB timeline ledger schema is invalid");
        }
        require_exact_object_fields(timeline.at("source_video"), {"path", "sha256"}, "source video binding");
        require_exact_object_fields(timeline.at("conversion_receipt"), {"path", "sha256"}, "conversion receipt binding");
        require_exact_object_fields(timeline.at("renderer_source"), {"path", "sha256", "rgb_sha256"}, "renderer source binding");
        require_exact_object_fields(
            timeline.at("decoder"),
            {"binary", "binary_sha256", "command", "decoded_rgb_sha256", "frame_count", "fps", "exit_code"},
            "pinned FFmpeg decoder binding"
        );
        require_exact_object_fields(
            timeline.at("raw_rgb"),
            {"path", "sha256", "bytes", "pixel_format", "shape"},
            "raw RGB binding"
        );
        const std::uint64_t frame_bytes =
            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 3ULL;
        if (frame_bytes == 0 || frame_bytes > std::numeric_limits<std::size_t>::max() / expected_frames) {
            throw std::runtime_error("raw RGB dimensions overflow byte count");
        }
        const std::size_t expected_bytes = static_cast<std::size_t>(frame_bytes * expected_frames);
        const auto& raw_binding = timeline.at("raw_rgb");
        const auto& decoder_binding = timeline.at("decoder");
        if (
            timeline.at("fps") != 60
            || timeline.at("frame_count") != expected_frames
            || raw_binding.at("path") != raw_rgb_path
            || raw_binding.at("sha256") != expected_raw_sha256
            || raw_binding.at("bytes") != expected_bytes
            || raw_binding.at("pixel_format") != "rgb24"
            || raw_binding.at("shape") != nlohmann::json::array({expected_frames, height, width, 3})
            || decoder_binding.at("decoded_rgb_sha256") != expected_raw_sha256
            || decoder_binding.at("frame_count") != expected_frames
            || decoder_binding.at("fps") != 60
            || decoder_binding.at("exit_code") != 0
            || timeline.at("renderer_source").at("rgb_sha256") != expected_raw_sha256
            || raw_rgb.size() != expected_bytes
            || sha256_hex(raw_rgb.data(), raw_rgb.size()) != expected_raw_sha256
        ) {
            throw std::runtime_error("raw RGB shape, byte count, decoder, or provenance binding drifted");
        }
        const auto& frame_rows = timeline.at("frames");
        if (!frame_rows.is_array() || frame_rows.size() != expected_frames) {
            throw std::runtime_error("RGB timeline row count differs from canonical frame count");
        }
        for (unsigned int frame_index = 0; frame_index < expected_frames; ++frame_index) {
            const auto& row = frame_rows.at(frame_index);
            require_exact_object_fields(
                row,
                {"frame_index", "cfr_timestamp", "cfr_timestamp_us", "renderer_timestamp_us", "rgb_sha256"},
                "RGB timeline row"
            );
            require_exact_object_fields(row.at("cfr_timestamp"), {"numerator", "denominator"}, "CFR rational timestamp");
            const std::uint64_t timestamp_us =
                (static_cast<std::uint64_t>(frame_index) * 1000000ULL + 30ULL) / 60ULL;
            const auto frame_offset = static_cast<std::size_t>(frame_index) * static_cast<std::size_t>(frame_bytes);
            if (
                row.at("frame_index") != frame_index
                || row.at("cfr_timestamp").at("numerator") != frame_index
                || row.at("cfr_timestamp").at("denominator") != 60
                || row.at("cfr_timestamp_us") != timestamp_us
                || row.at("renderer_timestamp_us") != timestamp_us
                || row.at("rgb_sha256") != sha256_hex(raw_rgb.data() + frame_offset, static_cast<std::size_t>(frame_bytes))
            ) {
                throw std::runtime_error("RGB timeline index, timestamp, or per-frame hash drifted");
            }
        }

        cv::setNumThreads(thread_limit);
        if (cv::getNumThreads() != thread_limit) {
            throw std::runtime_error("OpenCV did not accept the frozen thread limit");
        }
        iris::Log::Init(false, false);
        log_initialized = true;
        configuration.Init(config_directory.c_str());
        if (
            !(file_identity(config_path) == config_identity_before)
            || file_sha256(config_path) != expected_config_sha256
        ) {
            throw std::runtime_error("IRIS configuration changed while Configuration::Init reopened it");
        }
        configuration.SetPatternDetectionStatus(true);
        configuration.SetFrameResizeEnabled(false);

        cv::Size frame_size(width, height);
        analyser = std::make_unique<iris::VideoAnalyser>(&configuration);
        analyser->RealTimeInit(frame_size);
        analyser_initialized = true;

        nlohmann::json rows = nlohmann::json::array();
        bool hazardous = false;
        bool warning = false;
        for (unsigned int frame_index = 0; frame_index < expected_frames; ++frame_index) {
            const auto frame_offset = static_cast<std::size_t>(frame_index) * static_cast<std::size_t>(frame_bytes);
            cv::Mat rgb24(height, width, CV_8UC3, const_cast<unsigned char*>(raw_rgb.data() + frame_offset));
            if (!rgb24.isContinuous()) {
                throw std::runtime_error("raw RGB frame is not contiguous at consumption boundary");
            }
            const std::string pre_analyse_rgb_sha256 = mat_sha256(rgb24);
            if (pre_analyse_rgb_sha256 != frame_rows.at(frame_index).at("rgb_sha256")) {
                throw std::runtime_error("raw RGB frame drifted immediately before consumption");
            }
            cv::Mat boundary_bgr;
            cv::cvtColor(rgb24, boundary_bgr, cv::COLOR_RGB2BGR);
            if (boundary_bgr.type() != CV_8UC3 || !boundary_bgr.isContinuous()) {
                boundary_bgr = boundary_bgr.clone();
            }
            const std::string pre_analyse_bgr_sha256 = mat_sha256(boundary_bgr);
            const unsigned long timestamp_ms =
                (static_cast<unsigned long>(frame_index) * 1000UL) / 60UL;
            iris::FrameData data(frame_index + 1, timestamp_ms);
            unsigned int native_frame_index = frame_index;
            analyser->AnalyseFrame(boundary_bgr, native_frame_index, data);
            const std::string post_analyse_bgr_sha256 = mat_sha256(boundary_bgr);
            if (pre_analyse_bgr_sha256 != post_analyse_bgr_sha256) {
                throw std::runtime_error("AnalyseFrame mutated BGR pixels after pre-consumption hash");
            }
            const int luminance_result = static_cast<int>(data.luminanceFrameResult);
            const int red_result = static_cast<int>(data.redFrameResult);
            const int pattern_result = static_cast<int>(data.patternFrameResult);
            hazardous = hazardous || luminance_result >= 2 || red_result >= 2 || pattern_result >= 1;
            warning = warning || luminance_result == 1 || red_result == 1;
            rows.push_back({
                {"frame_index", frame_index},
                {"iris_frame_number", data.Frame},
                {"cfr_timestamp", {{"numerator", frame_index}, {"denominator", 60}}},
                {"cfr_timestamp_us_rounded", (static_cast<std::uint64_t>(frame_index) * 1000000ULL + 30ULL) / 60ULL},
                {"renderer_timestamp_us", frame_rows.at(frame_index).at("renderer_timestamp_us")},
                {"iris_timestamp_ms", data.TimeStampVal},
                {"shape", {boundary_bgr.rows, boundary_bgr.cols, boundary_bgr.channels()}},
                {"rgb_pixel_format", "rgb24"},
                {"bgr_pixel_format", "CV_8UC3"},
                {"rgb_sha256", pre_analyse_rgb_sha256},
                {"pre_analyse_rgb_sha256", pre_analyse_rgb_sha256},
                {"pre_analyse_bgr_sha256", pre_analyse_bgr_sha256},
                {"post_analyse_bgr_sha256", post_analyse_bgr_sha256},
                {"native_frame_data", {
                    {"luminance_average", data.LuminanceAverage},
                    {"luminance_flash_area", data.LuminanceFlashArea},
                    {"average_luminance_diff", data.AverageLuminanceDiff},
                    {"average_luminance_diff_acc", data.AverageLuminanceDiffAcc},
                    {"red_average", data.RedAverage},
                    {"red_flash_area", data.RedFlashArea},
                    {"average_red_diff", data.AverageRedDiff},
                    {"average_red_diff_acc", data.AverageRedDiffAcc},
                    {"pattern_risk", data.PatternRisk},
                    {"luminance_transitions", data.LuminanceTransitions},
                    {"red_transitions", data.RedTransitions},
                    {"luminance_extended_fail_count", data.LuminanceExtendedFailCount},
                    {"red_extended_fail_count", data.RedExtendedFailCount},
                    {"luminance_result", {{"code", luminance_result}, {"name", flash_result_name(data.luminanceFrameResult)}}},
                    {"red_result", {{"code", red_result}, {"name", flash_result_name(data.redFrameResult)}}},
                    {"pattern_area", data.patternArea},
                    {"pattern_detected_lines", data.patternDetectedLines},
                    {"pattern_result", {{"code", pattern_result}, {"name", pattern_result_name(data.patternFrameResult)}}}
                }}
            });
        }
        if (
            !raw_rgb.unchanged()
            || !timeline_bytes.unchanged()
            || sha256_hex(raw_rgb.data(), raw_rgb.size()) != expected_raw_sha256
            || sha256_hex(timeline_bytes.data(), timeline_bytes.size()) != expected_timeline_sha256
        ) {
            throw std::runtime_error("direct RGB input changed while IRIS consumed it");
        }
        analyser->DeInit();
        analyser_initialized = false;
        analyser.reset();
        iris::Log::ShutDown();
        log_initialized = false;

        const nlohmann::json payload = {
            {"schema", "flashpatch-l7-ea-iris-source-child-adapter-v1"},
            {"identity", "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC"},
            {"source_revision", "d96978ac1107f3463b77f69a9c1b1ec5d45291a0"},
            {"adapter_source_sha256", FLASHPATCH_ADAPTER_SOURCE_SHA256},
            {"input", {
                {"raw_rgb", {{"path", raw_rgb_path}, {"sha256", expected_raw_sha256}, {"bytes", raw_rgb.size()}}},
                {"timeline", {{"path", timeline_path}, {"sha256", expected_timeline_sha256}}},
                {"source_video", timeline.at("source_video")},
                {"conversion_receipt", timeline.at("conversion_receipt")},
                {"renderer_source", timeline.at("renderer_source")}
            }},
            {"configuration", {
                {"path", config_path},
                {"sha256", expected_config_sha256},
                {"overrides", {{"pattern_detection", true}, {"frame_resize", false}}}
            }},
            {"decoder", {
                {"api", "pinned_ffmpeg_raw_rgb24_prematerialization"},
                {"backend", "FFMPEG_COMMAND_BOUND_IN_TIMELINE"},
                {"binary", decoder_binding.at("binary")},
                {"binary_sha256", decoder_binding.at("binary_sha256")},
                {"command", decoder_binding.at("command")},
                {"reported_fps", 60},
                {"reported_frame_count", expected_frames},
                {"required_cfr", {{"numerator", 60}, {"denominator", 1}}},
                {"decoded_pixel_format", "rgb24"},
                {"analysis_thread_limit", thread_limit},
                {"analysis_threads_observed", cv::getNumThreads()},
                {"decoder_thread_control", "UNSUPPORTED_NOT_VERIFIED"},
                {"runtime_boundary", "FFMPEG_PREMATERIALIZATION_EXCLUDED_FROM_MEASURED_CHILD"}
            }},
            {"api_sequence", {
                "iris::Configuration::Init",
                "iris::VideoAnalyser::RealTimeInit",
                "iris::VideoAnalyser::AnalyseFrame",
                "iris::VideoAnalyser::DeInit"
            }},
            {"frame_count", expected_frames},
            {"frames", rows},
            {"prediction", hazardous ? "HAZARDOUS" : "SAFE"},
            {"warning", warning},
            {"runtime_timing_eligible", false},
            {"scoreable", false},
            {"scoreable_blockers", {
                "decoder_thread_control_unsupported_not_verified",
                "decoder_prematerialization_excluded_from_runtime_boundary",
                "local_execution_witness_not_independent",
                "independent_gold_receipt_missing",
                "frozen_public_case_ledger_missing"
            }}
        };
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create adapter output");
        }
        output << payload.dump(2) << '\n';
        output.close();
        if (!output) {
            throw std::runtime_error("cannot finalize adapter output");
        }
        return 0;
    } catch (const std::exception& error) {
        if (analyser_initialized && analyser) {
            analyser->DeInit();
            analyser_initialized = false;
        }
        analyser.reset();
        if (log_initialized) {
            iris::Log::ShutDown();
        }
        std::cerr << "EA IRIS source adapter error: " << error.what() << '\n';
        return 1;
    }
}
'''.strip()

_CENSUS_ENTRY_FIELDS = {
    "name",
    "repository_url",
    "revision",
    "source_checkout",
    "license",
    "license_artifact",
    "license_sha256",
    "distribution",
    "distribution_revision",
    "distribution_source_revision",
    "release_asset_sha256",
    "distribution_artifact",
    "distribution_sha256",
    "binary_artifact",
    "binary_sha256",
    "configuration_artifact",
    "configuration_sha256",
    "environment_artifact",
    "environment_sha256",
    "command_artifact",
    "command_sha256",
    "participant_conformance_artifact",
    "participant_conformance_sha256",
    "capability",
    "lane",
    "execution_status",
    "unscorable_reason",
}
_CENSUS_REQUIRED_HASH_FIELDS = {
    "license_sha256",
    "distribution_sha256",
}
_CENSUS_EXECUTION_HASH_FIELDS = {
    "binary_sha256",
    "configuration_sha256",
    "environment_sha256",
    "command_sha256",
}
_CENSUS_REQUIRED_ARTIFACT_HASHES = {
    "license_artifact": "license_sha256",
    "distribution_artifact": "distribution_sha256",
}
_CENSUS_EXECUTION_ARTIFACT_HASHES = {
    "binary_artifact": "binary_sha256",
    "configuration_artifact": "configuration_sha256",
    "environment_artifact": "environment_sha256",
    "command_artifact": "command_sha256",
}
_CENSUS_EXPECTED_PROVENANCE = {
    "FlashPatch": {
        "repository_url": "https://github.com/sergiobuilds/flashpatch",
        "license": "Apache-2.0",
        "capability": "detector",
        "lane": "direct-detector",
    },
    EA_IRIS_RELEASE_ORACLE_ID: {
        "repository_url": "https://github.com/electronicarts/IRIS",
        "revision": "fd3e09e4e6fce30a5141ad6eca94a4ff61096e05",
        "license": "BSD-3-Clause",
        "capability": "conformance-oracle",
        "lane": "conformance-oracle",
    },
    EA_IRIS_SOURCE_ADAPTER_ID: {
        "repository_url": "https://github.com/electronicarts/IRIS",
        "revision": "d96978ac1107f3463b77f69a9c1b1ec5d45291a0",
        "license": "BSD-3-Clause",
        "capability": "excluded-baseline",
        "lane": "excluded-semantic-mismatch",
    },
    KAYA_DIRECT_PARTICIPANT_ID: {
        "repository_url": KAYA_REPOSITORY_URL,
        "revision": KAYA_SOURCE_REVISION,
        "license": "BSD-3-Clause",
        "capability": "detector",
        "lane": "direct-detector",
    },
    "TooFlashy": {
        "repository_url": "https://github.com/hashb/TooFlashy",
        "revision": "8274e1ea09bd6099d384056f0fcb6fbc32cf0e3f",
        "license": "Apache-2.0",
        "capability": "detector",
        "lane": "direct-detector",
    },
    "EPI-LENS": {
        "repository_url": "https://github.com/Pi-0r-Tau/EPI-LENS",
        "revision": "a7c5ab95278e9c590324d6cb95b5f90982561f13",
        "license": "MIT",
        "capability": "detector",
        "lane": "reserve-detector",
    },
    "FFmpeg vf_photosensitivity": {
        "repository_url": "https://github.com/FFmpeg/FFmpeg",
        "revision": "601d9ee881fbd9d9ff44466c561c480ff244eb9f",
        "license": "GPL-3.0-or-later",
        "capability": "mitigation",
        "lane": "mitigation",
    },
}


@dataclass(frozen=True)
class ComparatorSpec:
    """Pinned executable contract for one independently maintained tool."""

    name: str
    repository_url: str
    revision: str
    license: str
    mode: str
    command: tuple[str, ...]
    raw_output_mode: str = "file"
    working_directory: Path | None = None
    expected_exit_codes: tuple[int, ...] = (0,)
    timeout_seconds: int = 120
    source_checkout: Path | None = None
    distribution: str | None = None
    distribution_revision: str | None = None
    configuration_sha256: str | None = None
    environment_sha256: str | None = None


@dataclass(frozen=True)
class IrisReleaseSpec:
    """Immutable provenance for an official EA IRIS application release."""

    repository_url: str
    source_revision: str
    release_tag: str
    release_asset: Path
    release_asset_sha256: str
    executable: Path
    appsettings: Path
    expected_fps: int


@dataclass(frozen=True)
class IrisSourceBuildSpec:
    """Frozen local build inputs for the d96978ac source comparator.

    The source checkout remains byte-for-byte upstream.  The generated adapter
    is compiled beside it and calls only the public real-time API.  Debian
    archives and the extracted dependency root are inputs, never inferred from
    whichever headers or libraries happen to be installed on the host.
    """

    source_checkout: Path
    dependency_root: Path
    dependency_archives: tuple[Path, ...]
    compiler: Path = Path("/usr/bin/g++")
    archiver: Path = Path("/usr/bin/ar")
    dpkg_deb: Path = Path("/usr/bin/dpkg-deb")
    readelf: Path = Path("/usr/bin/readelf")
    ldd: Path = Path("/usr/bin/ldd")
    minimal_media_receipt: Path | None = None


@dataclass(frozen=True)
class IrisMinimalMediaBuildSpec:
    """Auditable outputs of the pinned minimal OpenCV/FFmpeg source build.

    The source archives and exact build evidence are frozen independently from
    the IRIS build.  ``sdk_root`` contains only compile headers, the five
    source-required OpenCV modules, the four FFmpeg libraries, and frozen
    support libraries.  It is copied into the relocatable IRIS bundle; host
    libraries outside the narrow system ABI are never accepted.
    """

    opencv_archive: Path
    ffmpeg_archive: Path
    opencv_source: Path
    ffmpeg_source: Path
    opencv_build: Path
    opencv_install: Path
    ffmpeg_install: Path
    support_root: Path
    support_archives: tuple[Path, ...]
    build_tool_archives: tuple[Path, ...]
    sdk_root: Path
    ffmpeg_configure_command: tuple[str, ...]
    opencv_cmake_command: tuple[str, ...]
    compiler: Path = Path("/usr/bin/x86_64-linux-gnu-g++-13")
    readelf: Path = Path("/usr/bin/x86_64-linux-gnu-readelf")
    dpkg_deb: Path = Path("/usr/bin/dpkg-deb")


@dataclass(frozen=True)
class FairRuntimeProtocol:
    """Frozen equal-budget contract shared by every L7 detector runner.

    The protocol records policy; a repeat receipt must still prove that every
    scheduled run used it.  Installation, checkout verification and other
    untimed preparation remain outside the declared boundary.
    """

    machine_id: str
    operating_system: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    cpu_affinity: tuple[int, ...]
    thread_limit: int
    gpu_policy: str
    gpu_device: str | None
    gpu_isolation: str
    cache_policy: str
    concurrency_limit: int
    concurrency_lock_path: str
    process_isolation: str
    timeout_seconds: int
    repeats_required: int = 3
    retry_policy: str = "NO_RETRY"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def freeze_fair_runtime_protocol(protocol: FairRuntimeProtocol) -> dict[str, object]:
    """Validate and canonicalize one L7 equal-budget runtime protocol."""
    text_fields = {
        "machine_id": protocol.machine_id,
        "operating_system": protocol.operating_system,
        "architecture": protocol.architecture,
        "cpu_model": protocol.cpu_model,
        "cache_policy": protocol.cache_policy,
        "concurrency_lock_path": protocol.concurrency_lock_path,
        "process_isolation": protocol.process_isolation,
        "retry_policy": protocol.retry_policy,
    }
    if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
        raise ExternalLeagueError("fair runtime protocol text fields must be non-empty")
    if (
        isinstance(protocol.logical_cpu_count, bool)
        or not isinstance(protocol.logical_cpu_count, int)
        or protocol.logical_cpu_count <= 0
    ):
        raise ExternalLeagueError("fair runtime protocol logical CPU count is invalid")
    if not isinstance(protocol.cpu_affinity, (tuple, list)):
        raise ExternalLeagueError("fair runtime protocol CPU affinity is invalid")
    affinity = list(protocol.cpu_affinity)
    if (
        not affinity
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 or cpu >= protocol.logical_cpu_count for cpu in affinity)
        or len(set(affinity)) != len(affinity)
    ):
        raise ExternalLeagueError("fair runtime protocol CPU affinity is invalid")
    if (
        isinstance(protocol.thread_limit, bool)
        or not isinstance(protocol.thread_limit, int)
        or protocol.thread_limit <= 0
        or protocol.thread_limit > len(affinity)
    ):
        raise ExternalLeagueError("fair runtime protocol thread limit is invalid")
    if (
        protocol.gpu_policy != "DISABLED"
        or protocol.gpu_device is not None
        or protocol.gpu_isolation not in {"BWRAP_EMPTY_DEV", "DOCKER_EMPTY_DEV"}
    ):
        raise ExternalLeagueError("L7 fair runtime currently supports only enforced CPU-only execution")
    if protocol.cache_policy != "WARM_INPUT_PRETOUCHED":
        raise ExternalLeagueError("L7 fair runtime requires a verifiable warm-input cache policy")
    if isinstance(protocol.concurrency_limit, bool) or protocol.concurrency_limit != 1:
        raise ExternalLeagueError("L7 fair runtime requires concurrency limit one")
    lock_path = Path(protocol.concurrency_lock_path)
    if not lock_path.is_absolute():
        raise ExternalLeagueError("fair runtime concurrency lock path must be absolute")
    if protocol.process_isolation != "FRESH_SUBPROCESS_PER_REPEAT":
        raise ExternalLeagueError("L7 fair runtime process isolation policy is invalid")
    if (
        isinstance(protocol.timeout_seconds, bool)
        or not isinstance(protocol.timeout_seconds, int)
        or protocol.timeout_seconds <= 0
    ):
        raise ExternalLeagueError("fair runtime timeout budget is invalid")
    if protocol.repeats_required != 3 or protocol.retry_policy != "NO_RETRY":
        raise ExternalLeagueError("L7 fair runtime requires three scheduled runs and no retries")
    return {
        "schema": FAIR_RUNTIME_PROTOCOL_SCHEMA,
        "measurement_boundary": dict(FAIR_RUNTIME_BOUNDARY),
        "effective_environment_policy": dict(FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY),
        "machine": {
            "id": protocol.machine_id,
            "operating_system": protocol.operating_system,
            "architecture": protocol.architecture,
        },
        "cpu": {
            "model": protocol.cpu_model,
            "logical_count": protocol.logical_cpu_count,
            "affinity": affinity,
        },
        "threads": {"limit": protocol.thread_limit},
        "gpu": {
            "policy": protocol.gpu_policy,
            "device": protocol.gpu_device,
            "isolation": protocol.gpu_isolation,
        },
        "cache": {"policy": protocol.cache_policy},
        "concurrency": {
            "limit": protocol.concurrency_limit,
            "lock_path": str(lock_path.resolve()),
            "process_isolation": protocol.process_isolation,
        },
        "budget": {
            "timeout_seconds": protocol.timeout_seconds,
            "scheduled_repeats": protocol.repeats_required,
            "retry_policy": protocol.retry_policy,
        },
    }


def _validate_frozen_runtime_protocol(payload: object) -> dict[str, object]:
    """Re-open an already serialized protocol without trusting its hash."""
    if not isinstance(payload, Mapping) or payload.get("schema") != FAIR_RUNTIME_PROTOCOL_SCHEMA:
        raise ExternalLeagueError("fair runtime protocol schema is invalid")
    required = {
        "schema",
        "measurement_boundary",
        "effective_environment_policy",
        "machine",
        "cpu",
        "threads",
        "gpu",
        "cache",
        "concurrency",
        "budget",
    }
    if (
        set(payload) != required
        or payload.get("measurement_boundary") != FAIR_RUNTIME_BOUNDARY
        or payload.get("effective_environment_policy") != FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY
    ):
        raise ExternalLeagueError("fair runtime measurement boundary is invalid")
    machine = payload.get("machine")
    cpu = payload.get("cpu")
    threads = payload.get("threads")
    gpu = payload.get("gpu")
    cache = payload.get("cache")
    concurrency = payload.get("concurrency")
    budget = payload.get("budget")
    if not all(isinstance(item, Mapping) for item in (machine, cpu, threads, gpu, cache, concurrency, budget)):
        raise ExternalLeagueError("fair runtime environment or budget policy is invalid")
    try:
        rebuilt = FairRuntimeProtocol(
            machine_id=machine["id"],
            operating_system=machine["operating_system"],
            architecture=machine["architecture"],
            cpu_model=cpu["model"],
            logical_cpu_count=cpu["logical_count"],
            cpu_affinity=tuple(cpu["affinity"]),
            thread_limit=threads["limit"],
            gpu_policy=gpu["policy"],
            gpu_device=gpu["device"],
            gpu_isolation=gpu["isolation"],
            cache_policy=cache["policy"],
            concurrency_limit=concurrency["limit"],
            concurrency_lock_path=concurrency["lock_path"],
            process_isolation=concurrency["process_isolation"],
            timeout_seconds=budget["timeout_seconds"],
            repeats_required=budget["scheduled_repeats"],
            retry_policy=budget["retry_policy"],
        )
    except (KeyError, TypeError) as exc:
        raise ExternalLeagueError("fair runtime protocol fields are invalid") from exc
    frozen = freeze_fair_runtime_protocol(rebuilt)
    if dict(payload) != frozen:
        raise ExternalLeagueError("fair runtime protocol contains unrecognized policy fields")
    return frozen


def _freeze_runtime_protocol_input(
    protocol: FairRuntimeProtocol | Mapping[str, object] | None,
) -> dict[str, object] | None:
    if protocol is None:
        return None
    if isinstance(protocol, FairRuntimeProtocol):
        return freeze_fair_runtime_protocol(protocol)
    return _validate_frozen_runtime_protocol(protocol)


def _deterministic_schedule_order(
    comparators: Sequence[str],
    seed: int,
) -> list[str]:
    return sorted(
        comparators,
        key=lambda comparator: _sha256_bytes(f"{seed}\0{comparator}".encode("utf-8")),
    )


def _deterministic_schedule_slots(
    participants: Sequence[str],
    seed: int,
) -> list[dict[str, object]]:
    base_order = _deterministic_schedule_order(participants, seed)
    slots: list[dict[str, object]] = []
    slot_ordinal = 1
    for round_ordinal in range(1, 4):
        offset = (round_ordinal - 1) % len(base_order)
        round_order = base_order[offset:] + base_order[:offset]
        for position, comparator in enumerate(round_order, start=1):
            slots.append({
                "slot": slot_ordinal,
                "round": round_ordinal,
                "position": position,
                "comparator": comparator,
                "repeat_ordinal": round_ordinal,
            })
            slot_ordinal += 1
    return slots


def freeze_fair_runtime_schedule(
    comparators: Sequence[str],
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    input_sha256: str,
    *,
    seed: int,
) -> dict[str, object]:
    """Pre-register a deterministic, balanced three-repeat execution order."""
    if (
        not isinstance(comparators, Sequence)
        or isinstance(comparators, (str, bytes))
        or len(comparators) < 2
        or any(not isinstance(name, str) or not name for name in comparators)
        or len(set(comparators)) != len(comparators)
    ):
        raise ExternalLeagueError("fair runtime schedule requires at least two unique comparators")
    participants = sorted(comparators)
    if not set(participants).issubset(DIRECT_DETECTOR_POPULATION):
        raise ExternalLeagueError("fair runtime schedule supports only the direct detector population")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExternalLeagueError("fair runtime schedule seed must be a non-negative integer")
    if not isinstance(input_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
        raise ExternalLeagueError("fair runtime schedule input identity is invalid")
    frozen = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen is None:
        raise ExternalLeagueError("fair runtime schedule requires a frozen runtime protocol")
    slots = _deterministic_schedule_slots(participants, seed)
    return {
        "schema": FAIR_RUNTIME_SCHEDULE_SCHEMA,
        "policy": {
            "freeze_state": "PRE_FROZEN",
            "algorithm": "SHA256_SEEDED_CYCLIC_ROTATION_V1",
            "seed": seed,
            "repeats_per_comparator": 3,
            "one_process_at_a_time": True,
        },
        "participants": participants,
        "protocol_sha256": _canonical_json_sha256(frozen),
        "input_sha256": input_sha256,
        "slots": slots,
    }


def _validate_frozen_runtime_schedule(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or payload.get("schema") != FAIR_RUNTIME_SCHEDULE_SCHEMA:
        raise ExternalLeagueError("fair runtime schedule schema is invalid")
    required = {"schema", "policy", "participants", "protocol_sha256", "input_sha256", "slots"}
    if set(payload) != required:
        raise ExternalLeagueError("fair runtime schedule contains unrecognized fields")
    policy = payload.get("policy")
    participants = payload.get("participants")
    if not isinstance(policy, Mapping) or set(policy) != {
        "freeze_state",
        "algorithm",
        "seed",
        "repeats_per_comparator",
        "one_process_at_a_time",
    }:
        raise ExternalLeagueError("fair runtime schedule policy is invalid")
    if (
        policy.get("freeze_state") != "PRE_FROZEN"
        or policy.get("algorithm") != "SHA256_SEEDED_CYCLIC_ROTATION_V1"
        or policy.get("repeats_per_comparator") != 3
        or policy.get("one_process_at_a_time") is not True
    ):
        raise ExternalLeagueError("fair runtime schedule policy is invalid")
    if (
        not isinstance(participants, list)
        or len(participants) < 2
        or not all(isinstance(name, str) and name for name in participants)
    ):
        raise ExternalLeagueError("fair runtime schedule participant population is invalid")
    if (
        participants != sorted(participants)
        or len(set(participants)) != len(participants)
        or not set(participants).issubset(DIRECT_DETECTOR_POPULATION)
    ):
        raise ExternalLeagueError("fair runtime schedule participant population is invalid")
    for hash_field in ("protocol_sha256", "input_sha256"):
        value = payload.get(hash_field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ExternalLeagueError(f"fair runtime schedule {hash_field} is invalid")
    seed = policy.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExternalLeagueError("fair runtime schedule seed is invalid")
    expected = {
        "schema": FAIR_RUNTIME_SCHEDULE_SCHEMA,
        "policy": {
            "freeze_state": "PRE_FROZEN",
            "algorithm": "SHA256_SEEDED_CYCLIC_ROTATION_V1",
            "seed": seed,
            "repeats_per_comparator": 3,
            "one_process_at_a_time": True,
        },
        "participants": participants,
        "protocol_sha256": payload["protocol_sha256"],
        "input_sha256": payload["input_sha256"],
        "slots": _deterministic_schedule_slots(participants, seed),
    }
    if dict(payload) != expected:
        raise ExternalLeagueError("fair runtime schedule is not deterministic and balanced")
    positions: dict[str, set[int]] = {name: set() for name in participants}
    first_counts = {name: 0 for name in participants}
    last_counts = {name: 0 for name in participants}
    for entry in payload["slots"]:
        comparator = str(entry["comparator"])
        position = int(entry["position"])
        positions[comparator].add(position)
        if position == 1:
            first_counts[comparator] += 1
        if position == len(participants):
            last_counts[comparator] += 1
    if (
        any(len(values) < 2 for values in positions.values())
        or max(first_counts.values()) - min(first_counts.values()) > 1
        or max(last_counts.values()) - min(last_counts.values()) > 1
        or any(count == 3 for count in first_counts.values())
        or any(count == 3 for count in last_counts.values())
    ):
        raise ExternalLeagueError("fair runtime schedule position balance is invalid")
    return dict(payload)


def write_fair_runtime_schedule(
    comparators: Sequence[str],
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    input_sha256: str,
    schedule_path: Path | str,
    *,
    seed: int,
) -> dict[str, object]:
    """Write the schedule once so runs can bind the same pre-existing bytes."""
    destination = Path(schedule_path).resolve()
    if destination.exists():
        raise FileExistsError(f"fair runtime schedule already exists: {destination}")
    schedule = freeze_fair_runtime_schedule(
        comparators,
        runtime_protocol,
        input_sha256,
        seed=seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        **schedule,
        "receipt": str(destination),
        "artifact_sha256": _sha256_file(destination),
        "schedule_sha256": _canonical_json_sha256(schedule),
    }


def _load_schedule_assignment(
    schedule_receipt: Path | str | None,
    *,
    schedule_slot: int | None,
    protocol: Mapping[str, object] | None,
    comparator: str,
    repeat_ordinal: int | None,
    input_sha256: str,
) -> dict[str, object] | None:
    if schedule_receipt is None and schedule_slot is None:
        return None
    if schedule_receipt is None or schedule_slot is None or protocol is None:
        raise ExternalLeagueError("scheduled fair runtime requires schedule, slot, and protocol together")
    if isinstance(schedule_slot, bool) or not isinstance(schedule_slot, int):
        raise ExternalLeagueError("fair runtime schedule slot is invalid")
    path = Path(schedule_receipt).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("fair runtime schedule is unreadable") from exc
    schedule = _validate_frozen_runtime_schedule(payload)
    if schedule["protocol_sha256"] != _canonical_json_sha256(protocol):
        raise ExternalLeagueError("fair runtime schedule protocol binding mismatches")
    if schedule["input_sha256"] != input_sha256:
        raise ExternalLeagueError("fair runtime schedule input binding mismatches")
    entries = [entry for entry in schedule["slots"] if entry["slot"] == schedule_slot]
    if len(entries) != 1:
        raise ExternalLeagueError("fair runtime schedule slot is absent or duplicated")
    entry = entries[0]
    if entry["comparator"] != comparator or entry["repeat_ordinal"] != repeat_ordinal:
        raise ExternalLeagueError("fair runtime schedule assignment mismatches run identity")
    observed_stat = path.stat()
    return {
        "path": str(path),
        "artifact_sha256": _sha256_file(path),
        "schedule_sha256": _canonical_json_sha256(schedule),
        "stat": {
            "device": observed_stat.st_dev,
            "inode": observed_stat.st_ino,
            "size": observed_stat.st_size,
            "mtime_ns": observed_stat.st_mtime_ns,
            "ctime_ns": observed_stat.st_ctime_ns,
        },
        **entry,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _host_runtime_observation() -> dict[str, object]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise ExternalLeagueError("fair runtime CPU affinity cannot be observed") from exc
    logical_count = os.cpu_count()
    if not isinstance(logical_count, int) or logical_count <= 0:
        raise ExternalLeagueError("fair runtime logical CPU count cannot be observed")
    return {
        "machine": {
            "id": socket.gethostname(),
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_count": logical_count,
            "available_affinity": affinity,
        },
    }


def _inside_docker() -> bool:
    """Return whether this process has Docker's concrete container marker."""
    return Path("/.dockerenv").is_file()


def _visible_device_nodes() -> list[str]:
    """Record every device node visible to the executing process."""
    nodes: list[str] = []
    for device_root, _directory_names, file_names in os.walk("/dev", followlinks=False):
        for file_name in file_names:
            device_path = os.path.join(device_root, file_name)
            try:
                mode = os.stat(device_path, follow_symlinks=False).st_mode
            except OSError:
                continue
            if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
                nodes.append(device_path)
    return sorted(nodes)


def capture_fair_runtime_protocol(
    *,
    concurrency_lock_path: Path | str,
    timeout_seconds: int,
    cpu_affinity: Sequence[int] | None = None,
    thread_limit: int = 1,
    gpu_isolation: str = "BWRAP_EMPTY_DEV",
) -> FairRuntimeProtocol:
    """Capture a host-verifiable CPU-only protocol for later frozen runs."""
    if gpu_isolation == "DOCKER_EMPTY_DEV" and not _inside_docker():
        raise ExternalLeagueError("Docker empty-device isolation must be profiled inside Docker")
    observed = _host_runtime_observation()
    available = observed["cpu"]["available_affinity"]
    affinity = tuple(available if cpu_affinity is None else cpu_affinity)
    return FairRuntimeProtocol(
        machine_id=str(observed["machine"]["id"]),
        operating_system=str(observed["machine"]["operating_system"]),
        architecture=str(observed["machine"]["architecture"]),
        cpu_model=str(observed["cpu"]["model"]),
        logical_cpu_count=int(observed["cpu"]["logical_count"]),
        cpu_affinity=affinity,
        thread_limit=thread_limit,
        gpu_policy="DISABLED",
        gpu_device=None,
        gpu_isolation=gpu_isolation,
        cache_policy="WARM_INPUT_PRETOUCHED",
        concurrency_limit=1,
        concurrency_lock_path=str(Path(concurrency_lock_path).resolve()),
        process_isolation="FRESH_SUBPROCESS_PER_REPEAT",
        timeout_seconds=timeout_seconds,
    )


def _runtime_policy_environment(protocol: Mapping[str, object]) -> dict[str, str]:
    thread_limit = str(protocol["threads"]["limit"])
    environment = {
        "OMP_NUM_THREADS": thread_limit,
        "OPENBLAS_NUM_THREADS": thread_limit,
        "MKL_NUM_THREADS": thread_limit,
        "NUMEXPR_NUM_THREADS": thread_limit,
    }
    if protocol["gpu"]["policy"] == "DISABLED":
        environment.update({
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        })
    return environment


def _canonical_fair_base_environment(
    base_environment: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "PYTHONHASHSEED": "0",
        "MALLOC_ARENA_MAX": "1",
    }
    if base_environment is not None and "UV_PROJECT" in base_environment:
        uv_project = base_environment["UV_PROJECT"]
        if not isinstance(uv_project, str) or not Path(uv_project).resolve().is_dir():
            raise ExternalLeagueError("fair runtime UV project identity is invalid")
        environment["UV_PROJECT"] = str(Path(uv_project).resolve())
    return environment


def _runtime_schedule_environment(
    binding: Mapping[str, object] | None,
) -> dict[str, str]:
    if binding is None:
        return {}
    return {
        "FLASHPATCH_L7_SCHEDULE_SHA256": str(binding["schedule_sha256"]),
        "FLASHPATCH_L7_SCHEDULE_SLOT": str(binding["slot"]),
        "FLASHPATCH_L7_SCHEDULE_ROUND": str(binding["round"]),
        "FLASHPATCH_L7_SCHEDULE_POSITION": str(binding["position"]),
        "FLASHPATCH_L7_SCHEDULE_COMPARATOR": str(binding["comparator"]),
        "FLASHPATCH_L7_SCHEDULE_REPEAT": str(binding["repeat_ordinal"]),
    }


@contextmanager
def _fair_execution_context(
    protocol: Mapping[str, object] | None,
    canonical_input: Path,
    *,
    base_environment: Mapping[str, str] | None,
    schedule_binding: Mapping[str, object] | None = None,
    launcher_cwd: Path | str | None = None,
):
    if protocol is None:
        yield {
            "command_prefix": [],
            "environment": dict(base_environment) if base_environment is not None else None,
            "observation": None,
            "canonical_input": str(canonical_input.resolve()),
            "schedule_binding": None,
        }
        return
    frozen = _validate_frozen_runtime_protocol(protocol)
    observed = _host_runtime_observation()
    if observed["machine"] != frozen["machine"]:
        raise ExternalLeagueError("fair runtime machine identity differs from frozen protocol")
    if (
        observed["cpu"]["model"] != frozen["cpu"]["model"]
        or observed["cpu"]["logical_count"] != frozen["cpu"]["logical_count"]
        or not set(frozen["cpu"]["affinity"]).issubset(observed["cpu"]["available_affinity"])
    ):
        raise ExternalLeagueError("fair runtime CPU identity or available affinity differs from protocol")
    taskset_name = shutil.which("taskset")
    if taskset_name is None:
        raise ExternalLeagueError("fair runtime requires taskset affinity enforcement")
    taskset = Path(taskset_name).resolve()
    gpu_isolation = str(frozen["gpu"]["isolation"])
    bwrap: Path | None = None
    if gpu_isolation == "BWRAP_EMPTY_DEV":
        bwrap_name = shutil.which("bwrap")
        if bwrap_name is None:
            raise ExternalLeagueError("fair runtime requires bubblewrap GPU-device isolation")
        bwrap = Path(bwrap_name).resolve()
    elif gpu_isolation != "DOCKER_EMPTY_DEV":
        raise ExternalLeagueError("fair runtime GPU isolation policy is invalid")
    elif not _inside_docker():
        raise ExternalLeagueError("Docker empty-device isolation must execute inside Docker")
    lock_path = Path(str(frozen["concurrency"]["lock_path"]))
    if not lock_path.parent.is_dir():
        raise ExternalLeagueError("fair runtime concurrency lock parent is unavailable")
    environment = _canonical_fair_base_environment(base_environment)
    if launcher_cwd is not None:
        # PWD is deliberately excluded from effective-environment equality,
        # but it is separately recorded and must bind to the declared child
        # launcher directory.  Supplying it explicitly is necessary because
        # subprocess environments are intentionally constructed from scratch.
        launcher_path = Path(launcher_cwd).resolve()
        if not launcher_path.is_dir():
            raise ExternalLeagueError("fair runtime launcher directory is unavailable")
        environment["PWD"] = str(launcher_path)
    policy_environment = _runtime_policy_environment(frozen)
    environment.update(policy_environment)
    environment.update(_runtime_schedule_environment(schedule_binding))
    affinity_argument = ",".join(str(cpu) for cpu in frozen["cpu"]["affinity"])
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExternalLeagueError("fair runtime concurrency lock is already held") from exc
        pre_touch_sha256 = _sha256_file(canonical_input)
        observation = {
            **observed,
            "cpu": {
                **observed["cpu"],
                "taskset": {"path": str(taskset), "sha256": _sha256_file(taskset)},
            },
            "gpu": (
                {
                    "policy": "DISABLED",
                    "isolation": "BWRAP_EMPTY_DEV",
                    "bubblewrap": {"path": str(bwrap), "sha256": _sha256_file(bwrap)},
                }
                if bwrap is not None
                else {
                    "policy": "DISABLED",
                    "isolation": "DOCKER_EMPTY_DEV",
                    "docker": {
                        "container_marker_sha256": _sha256_file(Path("/.dockerenv")),
                        "visible_device_nodes": _visible_device_nodes(),
                    },
                }
            ),
            "cache": {
                "policy": "WARM_INPUT_PRETOUCHED",
                "input_sha256": pre_touch_sha256,
                "input_bytes": canonical_input.stat().st_size,
            },
            "concurrency": {
                "limit": 1,
                "lock_path": str(lock_path.resolve()),
                "lock_acquired": True,
            },
        }
        try:
            yield {
                "command_prefix": (
                    [
                        str(taskset), "--cpu-list", affinity_argument,
                        str(bwrap), "--bind", "/", "/", "--dev", "/dev",
                        "--proc", "/proc", "--die-with-parent",
                    ]
                    if bwrap is not None
                    else [str(taskset), "--cpu-list", affinity_argument]
                ),
                "environment": environment,
                "observation": observation,
                "canonical_input": str(canonical_input.resolve()),
                "schedule_binding": dict(schedule_binding) if schedule_binding is not None else None,
            }
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _instrument_fair_command(
    execution: Mapping[str, object],
    tool_command: Sequence[str],
    probe_path: Path,
) -> list[str]:
    prefix = execution.get("command_prefix")
    if not isinstance(prefix, list) or not all(isinstance(part, str) for part in prefix):
        return list(tool_command)
    return [
        *prefix,
        str(Path(sys.executable).resolve()),
        "-c",
        _RUNTIME_PROBE_SCRIPT,
        str(probe_path.resolve()),
        str(execution.get("canonical_input")),
        str(execution["schedule_binding"]["path"]) if isinstance(execution.get("schedule_binding"), Mapping) else "-",
        *tool_command,
    ]


def _load_child_runtime_probe(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("child runtime probe is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "flashpatch-l7-child-runtime-probe-v1":
        raise ExternalLeagueError("child runtime probe schema is invalid")
    return payload


def _runtime_environment_sha256(protocol: Mapping[str, object]) -> str:
    return _canonical_json_sha256({
        key: protocol[key]
        for key in (
            "effective_environment_policy",
            "machine",
            "cpu",
            "threads",
            "gpu",
            "cache",
            "concurrency",
        )
    })


def _normalized_terminal_identity(
    observation: Mapping[str, object] | None,
    *,
    normalizer: str,
) -> dict[str, object] | None:
    if not isinstance(observation, Mapping):
        return None
    return {
        "schema": "flashpatch-l7-normalized-terminal-observation-v1",
        "normalizer": normalizer,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "sha256": _canonical_json_sha256(dict(observation)),
    }


def _fair_runtime_run_receipt(
    protocol: Mapping[str, object] | None,
    *,
    comparator: str,
    scheduled_repeat_ordinal: int | None,
    schedule_binding: Mapping[str, object] | None,
    input_sha256: str,
    started_monotonic_ns: int,
    finished_monotonic_ns: int,
    wall_time_ns: int,
    timed_out: bool,
    observation: Mapping[str, object] | None,
    normalizer: str,
    observed_environment: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if protocol is None:
        return None
    frozen = _validate_frozen_runtime_protocol(protocol)
    if (
        isinstance(scheduled_repeat_ordinal, bool)
        or not isinstance(scheduled_repeat_ordinal, int)
        or scheduled_repeat_ordinal not in {1, 2, 3}
    ):
        raise ExternalLeagueError("fair runtime run requires a scheduled repeat ordinal from one through three")
    budget = frozen["budget"]
    if (
        isinstance(started_monotonic_ns, bool)
        or not isinstance(started_monotonic_ns, int)
        or started_monotonic_ns < 0
        or isinstance(finished_monotonic_ns, bool)
        or not isinstance(finished_monotonic_ns, int)
        or finished_monotonic_ns <= started_monotonic_ns
        or finished_monotonic_ns - started_monotonic_ns != wall_time_ns
    ):
        raise ExternalLeagueError("fair runtime monotonic interval is invalid")
    if schedule_binding is not None and (
        schedule_binding.get("comparator") != comparator
        or schedule_binding.get("repeat_ordinal") != scheduled_repeat_ordinal
    ):
        raise ExternalLeagueError("fair runtime schedule binding mismatches run receipt")
    return {
        "schema": FAIR_RUNTIME_RUN_SCHEMA,
        "protocol_sha256": _canonical_json_sha256(frozen),
        "measurement_boundary": frozen["measurement_boundary"],
        "environment_policy_sha256": _runtime_environment_sha256(frozen),
        "observed_environment": dict(observed_environment) if observed_environment is not None else None,
        "timeout_seconds": budget["timeout_seconds"],
        "scheduled_repeat_ordinal": scheduled_repeat_ordinal,
        "schedule_binding": dict(schedule_binding) if schedule_binding is not None else None,
        "attempt_ordinal": 1,
        "retry_count": 0,
        "retry_policy": "NO_RETRY",
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": finished_monotonic_ns,
        "wall_time_ns": wall_time_ns,
        "timed_out": timed_out,
        "input_identity_sha256": input_sha256,
        "normalized_terminal_observation": _normalized_terminal_identity(
            observation,
            normalizer=normalizer,
        ),
    }


def _normalized_repository_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix(".git").removesuffix("/")


_FLASHPATCH_EXECUTION_PATHS = (
    "src/flashpatch",
    "scripts/run_l7_score_pipeline.py",
    "scripts/build_census.py",
    "scripts/l7-external-host.Dockerfile",
)

_REMOTE_VERIFICATION_RECEIPT_SCHEMA = "flashpatch-l7-remote-verification-receipt-v1"


def _verified_remote_receipt(checkout: Path, remote_url: str, head: str) -> bool | None:
    """Use an explicit coordinator fetch receipt when an executor lacks Git auth.

    The receipt is not a substitute for an unverifiable local tracking ref.  It
    records the coordinator's direct ``ls-remote`` observation and is accepted
    only when its remote URL, checked-out head, and observed remote head are
    all exact matches.  Without this opt-in receipt, the caller still performs
    its own direct remote query.
    """
    raw_path = os.environ.get("FLASHPATCH_REMOTE_VERIFICATION_RECEIPT")
    if not raw_path:
        return None
    path = Path(raw_path)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("FlashPatch remote verification receipt is unavailable") from exc
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema", "remote_url", "head", "observed_remote_head"
    }:
        raise ExternalLeagueError("FlashPatch remote verification receipt is invalid")
    if (
        receipt.get("schema") != _REMOTE_VERIFICATION_RECEIPT_SCHEMA
        or not isinstance(receipt.get("remote_url"), str)
        or not isinstance(receipt.get("head"), str)
        or not isinstance(receipt.get("observed_remote_head"), str)
        or _normalized_repository_url(str(receipt["remote_url"])) != _normalized_repository_url(remote_url)
        or receipt["head"] != head
        or receipt["observed_remote_head"] != head
    ):
        raise ExternalLeagueError("FlashPatch remote verification receipt does not match the checkout")
    try:
        path.resolve(strict=True).relative_to(checkout.resolve(strict=True))
    except ValueError as exc:
        raise ExternalLeagueError("FlashPatch remote verification receipt escapes the checkout") from exc
    return True


def flashpatch_execution_provenance(checkout: Path | str | None = None) -> dict[str, object]:
    """Freeze executable L7 inputs without making prose commits new evidence.

    ``revision`` and ``tree`` are real Git identities of the checkout.  The
    separately named ``execution_revision`` is the SHA-1 digest of the pinned
    Git objects that can alter detector execution.  This prevents a code
    closure digest from being mistaken for a Git commit during re-opening.
    """
    checkout = Path(checkout or Path(__file__).resolve().parents[2]).resolve()
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            check=False,
            text=True,
        )

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    origin = git("remote", "get-url", "origin")
    status = git("status", "--porcelain", "--untracked-files=all", "--", *_FLASHPATCH_EXECUTION_PATHS)
    objects = [git("rev-parse", f"HEAD:{path}") for path in _FLASHPATCH_EXECUTION_PATHS]
    if any(result.returncode != 0 for result in (head, tree, origin, status, *objects)):
        raise ExternalLeagueError("FlashPatch source checkout provenance is unavailable")
    execution_revision = hashlib.sha1(
        json.dumps(
            {
                "paths": [
                    {"path": path, "git_object": result.stdout.strip()}
                    for path, result in zip(_FLASHPATCH_EXECUTION_PATHS, objects, strict=True)
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    # A local origin/master ref is a cache, not evidence that the current
    # revision was pushed.  It may be stale when pushing via an HTTPS override
    # while origin keeps an SSH fetch URL.  Query the intended remote directly.
    remote_url = os.environ.get("FLASHPATCH_REMOTE_URL", origin.stdout.strip())
    receipt_pushed = _verified_remote_receipt(checkout, remote_url, head.stdout.strip())
    if receipt_pushed is None:
        remote_head = subprocess.run(
            ["git", "ls-remote", "--exit-code", remote_url, "refs/heads/master"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        fields = remote_head.stdout.strip().split() if remote_head.returncode == 0 else []
        pushed = bool(fields and fields[0] == head.stdout.strip())
    else:
        pushed = receipt_pushed
    return {
        "revision": head.stdout.strip(),
        "tree": tree.stdout.strip(),
        "execution_revision": execution_revision,
        "head": head.stdout.strip(),
        "origin": _normalized_repository_url(origin.stdout),
        "clean": status.stdout == "",
        "pushed": pushed,
        "remote_verification_url": _normalized_repository_url(remote_url),
    }


def _flashpatch_checkout_provenance() -> dict[str, object]:
    """Backward-compatible internal alias for the execution provenance gate."""
    return flashpatch_execution_provenance()


def _current_ffmpeg_build_provenance() -> dict[str, object]:
    binary = Path("/usr/bin/ffmpeg")
    package_query = Path("/usr/bin/dpkg-query")
    if not binary.is_file() or not package_query.is_file():
        raise ExternalLeagueError("the census-pinned FFmpeg distribution is unavailable")
    version = subprocess.run([str(binary), "-version"], capture_output=True, check=False)
    package = subprocess.run(
        [str(package_query), "-W", "-f=${Package}\t${Version}\t${Architecture}\n", "ffmpeg"],
        capture_output=True,
        check=False,
    )
    copyright_path = Path("/usr/share/doc/ffmpeg/copyright")
    if version.returncode != 0 or package.returncode != 0 or not copyright_path.is_file():
        raise ExternalLeagueError("FFmpeg package build provenance is unavailable")
    try:
        package_name, package_revision, architecture = package.stdout.decode("utf-8").strip().split("\t")
    except ValueError as exc:
        raise ExternalLeagueError("FFmpeg package identity is invalid") from exc
    return {
        "binary_sha256": _sha256_file(binary),
        "configuration_sha256": _sha256_bytes(version.stdout),
        "distribution_sha256": _sha256_bytes(package.stdout),
        "license_sha256": _sha256_file(copyright_path),
        "distribution": package_name,
        "distribution_revision": package_revision,
        "architecture": architecture,
    }


def _verify_upstream_checkout(entry: Mapping[str, object]) -> dict[str, object]:
    checkout_value = entry.get("source_checkout")
    if not isinstance(checkout_value, str) or not checkout_value:
        raise ExternalLeagueError(f"comparator source checkout is missing: {entry.get('name')}")
    checkout = Path(checkout_value).resolve()
    if not checkout.is_dir():
        raise ExternalLeagueError(f"comparator source checkout is unavailable: {entry.get('name')}")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            check=False,
            text=True,
        )

    head = git("rev-parse", "HEAD")
    origin = git("remote", "get-url", "origin")
    status = git("status", "--porcelain", "--untracked-files=no")
    tree = git("rev-parse", "HEAD^{tree}")
    if any(result.returncode != 0 for result in (head, origin, status, tree)):
        raise ExternalLeagueError(f"comparator source checkout cannot be verified: {entry.get('name')}")
    if head.stdout.strip() != entry.get("revision") or _normalized_repository_url(origin.stdout) != entry.get("repository_url") or status.stdout != "":
        raise ExternalLeagueError(f"comparator source checkout provenance mismatches: {entry.get('name')}")
    license_candidates = [
        path for path in checkout.iterdir()
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING"))
    ]
    if not any(_sha256_file(path) == entry.get("license_sha256") for path in license_candidates):
        raise ExternalLeagueError(f"comparator checkout license does not match census: {entry.get('name')}")
    return {
        "path": str(checkout),
        "revision": head.stdout.strip(),
        "tree": tree.stdout.strip(),
        "origin": _normalized_repository_url(origin.stdout),
        "clean": True,
    }


def _verify_iris_release_artifacts(entry: Mapping[str, object]) -> None:
    if (
        entry.get("distribution") != "official-ubuntu-example-app-1.1.0"
        or entry.get("distribution_revision") != "1.1.0"
        or entry.get("distribution_source_revision") != "fd3e09e4e6fce30a5141ad6eca94a4ff61096e05"
        or entry.get("release_asset_sha256") != "440eb0cb814a03a4eff7c8c4f499492b669a33cf2ba4f23843b479365eeedaeb"
        or entry.get("binary_sha256") != "a134ad3280bc8cb48bcade4a787ca2d4bd332abe3b3ba60b01d7a6eda5f203e0"
        or entry.get("configuration_sha256") != "dd4e601c362ddaab6314fcb56bc38327cfe438599c5ace81d65288ffaddd3d17"
    ):
        raise ExternalLeagueError("EA IRIS census does not match the pinned official 1.1.0 Ubuntu release")


def _verify_tooflashy_distribution(
    entry: Mapping[str, object],
    command_template: Sequence[object],
    environment: Mapping[str, str],
) -> None:
    uv = _uv_executable()
    expected_command = [str(uv), "run", "tooflashy", "--json", "{input}"]
    if (
        not uv.is_file()
        or entry.get("binary_sha256") != _sha256_file(uv)
        or list(command_template) != expected_command
        or set(environment) != {"UV_PROJECT", "PATH"}
        or environment.get("UV_PROJECT") != str(Path(str(entry.get("source_checkout"))).resolve())
        or environment.get("PATH") != "/usr/bin:/bin"
    ):
        raise ExternalLeagueError("TooFlashy census does not bind the pinned checkout through the canonical uv command")


def _resolve_census_artifact(root: Path, relative: object, *, name: str, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ExternalLeagueError(f"comparator artifact path is invalid: {name}:{field}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExternalLeagueError(f"comparator artifact escapes evidence root: {name}:{field}") from exc
    if not path.is_file():
        raise ExternalLeagueError(f"comparator provenance artifact is missing: {name}:{field}")
    return path


def validate_comparator_census(
    manifest: Mapping[str, object],
    *,
    artifact_root: Path | str,
) -> dict[str, object]:
    """Validate and hash-bind the L7 G1 comparator population.

    This gate freezes identities and capabilities only.  It cannot score a
    detector, rank detector and mitigation tools together, or authorize a win.
    Hashes bind the declared source, distribution, binary, configuration,
    environment and command evidence for later receipt verification.
    """
    cache_key = (_canonical_json_sha256(manifest), str(Path(artifact_root).resolve()))
    cached = _CENSUS_VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ExternalLeagueError("comparator census artifact root is unavailable")
    expected_manifest_fields = {
        "schema",
        "detector_population",
        "conformance_oracle_population",
        "excluded_semantic_mismatch_population",
        "mitigation_population",
        "reserve_detector_population",
        "comparators",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise ExternalLeagueError("comparator census manifest fields are invalid")
    if manifest.get("schema") != COMPARATOR_CENSUS_SCHEMA:
        raise ExternalLeagueError("comparator census manifest schema is invalid")

    detector_population = manifest.get("detector_population")
    conformance_oracle_population = manifest.get("conformance_oracle_population")
    excluded_semantic_mismatch_population = manifest.get("excluded_semantic_mismatch_population")
    mitigation_population = manifest.get("mitigation_population")
    reserve_population = manifest.get("reserve_detector_population")
    if detector_population != list(DIRECT_DETECTOR_POPULATION):
        raise ExternalLeagueError(
            "direct detector population must be FlashPatch, "
            f"{KAYA_DIRECT_PARTICIPANT_ID}, and TooFlashy"
        )
    if conformance_oracle_population != list(CONFORMANCE_ORACLE_POPULATION):
        raise ExternalLeagueError(
            f"conformance oracle population must contain only {EA_IRIS_RELEASE_ORACLE_ID}"
        )
    if excluded_semantic_mismatch_population != list(EXCLUDED_SEMANTIC_MISMATCH_POPULATION):
        raise ExternalLeagueError(
            "excluded semantic-mismatch population must contain only "
            f"{EA_IRIS_SOURCE_ADAPTER_ID}"
        )
    if mitigation_population != list(MITIGATION_POPULATION):
        raise ExternalLeagueError("mitigation population must contain only FFmpeg vf_photosensitivity")
    if reserve_population != list(RESERVE_DETECTOR_POPULATION):
        raise ExternalLeagueError("reserve detector population must contain only EPI-LENS")

    comparators = manifest.get("comparators")
    if not isinstance(comparators, list):
        raise ExternalLeagueError("comparator census entries must be a list")
    if any(isinstance(item, Mapping) and item.get("name") == "TooFlashy_or_EPI_LENS" for item in comparators):
        raise ExternalLeagueError("TooFlashy_or_EPI_LENS is an ambiguous comparator identity")
    expected_names = set(_CENSUS_EXPECTED_PROVENANCE)
    names = [item.get("name") for item in comparators if isinstance(item, Mapping)]
    if (
        len(comparators) != len(expected_names)
        or len(names) != len(comparators)
        or not all(isinstance(name, str) for name in names)
        or set(names) != expected_names
        or len(set(names)) != len(names)
    ):
        raise ExternalLeagueError("comparator census must contain each fixed identity exactly once")

    normalized: list[dict[str, object]] = []
    flashpatch_checkout = _flashpatch_checkout_provenance()
    if flashpatch_checkout["clean"] is not True or flashpatch_checkout["pushed"] is not True:
        raise ExternalLeagueError("FlashPatch census requires a clean checkout at a pushed revision")
    for raw_entry in comparators:
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != _CENSUS_ENTRY_FIELDS:
            raise ExternalLeagueError("comparator census entry fields are invalid")
        entry = dict(raw_entry)
        name = str(entry["name"])
        expected = _CENSUS_EXPECTED_PROVENANCE[name]
        if not isinstance(entry["repository_url"], str) or entry["repository_url"] != expected["repository_url"]:
            raise ExternalLeagueError(f"comparator repository provenance is invalid: {name}")
        if not isinstance(entry["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) is None:
            raise ExternalLeagueError(f"comparator revision is not a pinned commit: {name}")
        expected_revision = expected.get("revision")
        if expected_revision is not None and entry["revision"] != expected_revision:
            raise ExternalLeagueError(f"comparator revision differs from the frozen census: {name}")
        if name == "FlashPatch" and (
            entry["revision"] != flashpatch_checkout["revision"]
            or _normalized_repository_url(str(entry["repository_url"])) != flashpatch_checkout["origin"]
            or not isinstance(entry["source_checkout"], str)
            or Path(entry["source_checkout"]).resolve() != Path(__file__).resolve().parents[2]
        ):
            raise ExternalLeagueError("FlashPatch revision or origin differs from the executing checkout")
        source_checkout_provenance: dict[str, object] | None = None
        if name in {
            EA_IRIS_RELEASE_ORACLE_ID,
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
            "TooFlashy",
            "EPI-LENS",
        }:
            source_checkout_provenance = _verify_upstream_checkout(entry)
        elif name == "FFmpeg vf_photosensitivity" and entry["source_checkout"] is not None:
            raise ExternalLeagueError("FFmpeg package census must separate the installed build from the algorithm reference source")
        if entry["license"] != expected["license"]:
            if name == "FFmpeg vf_photosensitivity":
                raise ExternalLeagueError("FFmpeg distribution must declare effective GPL-3.0-or-later build provenance")
            raise ExternalLeagueError(f"comparator license provenance is invalid: {name}")
        if not isinstance(entry["distribution"], str) or not entry["distribution"].strip():
            raise ExternalLeagueError(f"comparator distribution provenance is missing: {name}")
        if not isinstance(entry["distribution_revision"], str) or not entry["distribution_revision"].strip():
            raise ExternalLeagueError(f"comparator distribution revision is missing: {name}")
        if not isinstance(entry["distribution_source_revision"], str) or not entry["distribution_source_revision"].strip():
            raise ExternalLeagueError(f"comparator distribution source revision is missing: {name}")
        if name == EA_IRIS_RELEASE_ORACLE_ID:
            if not isinstance(entry["release_asset_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["release_asset_sha256"]) is None:
                raise ExternalLeagueError("EA IRIS official release asset hash is invalid")
        elif entry["release_asset_sha256"] is not None:
            raise ExternalLeagueError(f"non-release comparator cannot declare an IRIS release asset: {name}")
        for field in _CENSUS_REQUIRED_HASH_FIELDS:
            value = entry[field]
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ExternalLeagueError(f"comparator provenance hash is invalid: {name}:{field}")
        execution_fields = {
            *(_CENSUS_EXECUTION_HASH_FIELDS),
            *(_CENSUS_EXECUTION_ARTIFACT_HASHES),
        }
        non_executable_census_identities = {
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
        }
        if name in non_executable_census_identities:
            if any(entry[field] is not None for field in execution_fields):
                raise ExternalLeagueError(
                    f"unscored or excluded comparator cannot declare executable census evidence: {name}"
                )
        else:
            for field in _CENSUS_EXECUTION_HASH_FIELDS:
                value = entry[field]
                if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise ExternalLeagueError(f"comparator provenance hash is invalid: {name}:{field}")
        artifacts: dict[str, Path] = {}
        artifact_contract = dict(_CENSUS_REQUIRED_ARTIFACT_HASHES)
        if name not in non_executable_census_identities:
            artifact_contract.update(_CENSUS_EXECUTION_ARTIFACT_HASHES)
        if name == KAYA_DIRECT_PARTICIPANT_ID:
            participant_hash = entry["participant_conformance_sha256"]
            if not isinstance(participant_hash, str) or re.fullmatch(r"[0-9a-f]{64}", participant_hash) is None:
                raise ExternalLeagueError("Kaya participant conformance receipt hash is invalid")
            artifact_contract["participant_conformance_artifact"] = "participant_conformance_sha256"
        elif (
            entry["participant_conformance_artifact"] is not None
            or entry["participant_conformance_sha256"] is not None
        ):
            raise ExternalLeagueError(
                f"non-Kaya comparator cannot declare a Kaya participant conformance receipt: {name}"
            )
        for artifact_field, hash_field in artifact_contract.items():
            artifact = _resolve_census_artifact(root, entry[artifact_field], name=name, field=artifact_field)
            if _sha256_file(artifact) != entry[hash_field]:
                raise ExternalLeagueError(f"comparator provenance artifact hash mismatches: {name}:{artifact_field}")
            artifacts[artifact_field] = artifact
        if len(set(artifacts.values())) != len(artifacts):
            raise ExternalLeagueError(f"comparator provenance roles must use distinct artifacts: {name}")
        if name not in non_executable_census_identities and not os.access(artifacts["binary_artifact"], os.X_OK):
            raise ExternalLeagueError(f"comparator binary artifact is not executable: {name}")
        try:
            distribution_evidence = json.loads(artifacts["distribution_artifact"].read_text(encoding="utf-8"))
            command_template = (
                json.loads(artifacts["command_artifact"].read_text(encoding="utf-8"))
                if name not in non_executable_census_identities
                else None
            )
            environment_evidence = (
                json.loads(artifacts["environment_artifact"].read_text(encoding="utf-8"))
                if name not in non_executable_census_identities
                else None
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError(f"comparator distribution, command, or environment artifact is invalid: {name}") from exc
        expected_distribution_evidence = {
            "schema": "flashpatch-comparator-distribution-provenance-v1",
            "name": name,
            "repository_url": entry["repository_url"],
            "revision": entry["revision"],
            "source_checkout": entry["source_checkout"],
            "license": entry["license"],
            "license_sha256": entry["license_sha256"],
            "distribution": entry["distribution"],
            "distribution_revision": entry["distribution_revision"],
            "distribution_source_revision": entry["distribution_source_revision"],
            "release_asset_sha256": entry["release_asset_sha256"],
            "binary_sha256": entry["binary_sha256"],
            "configuration_sha256": entry["configuration_sha256"],
            "environment_sha256": entry["environment_sha256"],
            "command_sha256": entry["command_sha256"],
            "participant_conformance_sha256": entry["participant_conformance_sha256"],
        }
        if distribution_evidence != expected_distribution_evidence:
            raise ExternalLeagueError(f"comparator distribution evidence does not bind its provenance: {name}")
        if name not in non_executable_census_identities:
            if not isinstance(command_template, list) or not command_template or not all(isinstance(part, str) and part for part in command_template):
                raise ExternalLeagueError(f"comparator command template is invalid: {name}")
            if not isinstance(environment_evidence, Mapping) or not all(isinstance(key, str) and isinstance(value, str) for key, value in environment_evidence.items()):
                raise ExternalLeagueError(f"comparator runtime environment is invalid: {name}")
        license_evidence = artifacts["license_artifact"].read_text(encoding="utf-8", errors="replace")
        license_markers = {
            "Apache-2.0": ("Apache License", "Version 2.0"),
            "BSD-3-Clause": ("Redistribution and use", "source and binary forms"),
            "MIT": ("Permission is hereby granted",),
        }
        if entry["license"] in license_markers and not all(marker in license_evidence for marker in license_markers[str(entry["license"])]):
            raise ExternalLeagueError(f"comparator license artifact does not match its SPDX declaration: {name}")
        if name == "FFmpeg vf_photosensitivity":
            configuration_evidence = artifacts["configuration_artifact"].read_text(encoding="utf-8", errors="replace")
            if "GPL-3+" not in license_evidence or "effectively licensed" not in license_evidence:
                raise ExternalLeagueError("FFmpeg package evidence does not prove effective GPL-3+ licensing")
            if "--enable-gpl" not in configuration_evidence:
                raise ExternalLeagueError("FFmpeg build configuration does not prove a GPL-enabled distribution")
            live_ffmpeg = _current_ffmpeg_build_provenance()
            for field in ("binary_sha256", "configuration_sha256", "license_sha256", "distribution", "distribution_revision"):
                if entry[field] != live_ffmpeg[field]:
                    raise ExternalLeagueError(f"FFmpeg artifact differs from the installed package build: {field}")
        elif name == EA_IRIS_RELEASE_ORACLE_ID:
            _verify_iris_release_artifacts(entry)
        elif name == "TooFlashy":
            _verify_tooflashy_distribution(entry, command_template, environment_evidence)
        if entry["capability"] != expected["capability"] or entry["lane"] != expected["lane"]:
            raise ExternalLeagueError(f"comparator capability or league lane is invalid: {name}")
        status = entry["execution_status"]
        reason = entry["unscorable_reason"]
        if status not in {"FIXED", "RUNNABLE", "UNSCORABLE"}:
            raise ExternalLeagueError(f"comparator execution status is invalid: {name}")
        if status == "UNSCORABLE":
            if not isinstance(reason, str) or not reason:
                raise ExternalLeagueError(f"UNSCORABLE comparator must record a reason: {name}")
        elif reason is not None:
            raise ExternalLeagueError(f"scoreable-state language is inconsistent with census status: {name}")
        if name == "EPI-LENS" and (
            status != "UNSCORABLE"
            or reason != "full_same_input_application_runner_missing"
        ):
            raise ExternalLeagueError("EPI-LENS remains UNSCORABLE until a full same-input application runner exists")
        if name == EA_IRIS_SOURCE_ADAPTER_ID and (
            status != "UNSCORABLE"
            or reason != "semantic_conformance_mismatch_excluded"
        ):
            raise ExternalLeagueError(
                "EA IRIS source adapter is an excluded semantic-mismatch baseline only"
            )
        participant_conformance: dict[str, object] | None = None
        if name == KAYA_DIRECT_PARTICIPANT_ID:
            if (
                status != "UNSCORABLE"
                or reason != "natural_corpus_gold_parity_and_fair_repeats_missing"
            ):
                raise ExternalLeagueError(
                    "Kaya can enter only as an UNSCORABLE participant while natural corpus, gold, parity, and fair repeats are missing"
                )
            verified = verify_kaya_participant_conformance_receipt(
                artifacts["participant_conformance_artifact"]
            )
            if (
                verified.get("status") != "VERIFIED"
                or verified.get("identity") != KAYA_DIRECT_PARTICIPANT_ID
                or verified.get("prototype_identity") != KAYA_PROTOTYPE_ID
                or verified.get("scoreable") is not False
                or verified.get("unscored_population_authorized") is not True
                or verified.get("external_claim_authorized") is not False
            ):
                raise ExternalLeagueError(
                    "Kaya participant conformance receipt cannot authorize the unscored population"
                )
            participant_conformance = {
                "schema": verified["schema"],
                "identity": verified["identity"],
                "prototype_identity": verified["prototype_identity"],
                "status": verified["status"],
                "scoreable": False,
                "receipt": str(artifacts["participant_conformance_artifact"]),
                "receipt_sha256": entry["participant_conformance_sha256"],
            }
        normalized.append({
            **entry,
            "source_checkout_provenance": source_checkout_provenance,
            "participant_conformance": participant_conformance,
            "provenance_sha256": _canonical_json_sha256(entry),
        })

    normalized.sort(key=lambda entry: str(entry["name"]))
    result = {
        "schema": COMPARATOR_CENSUS_RECEIPT_SCHEMA,
        "manifest_schema": COMPARATOR_CENSUS_SCHEMA,
        "manifest_sha256": _canonical_json_sha256(manifest),
        "artifact_root": str(root),
        "flashpatch_checkout": flashpatch_checkout,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "conformance_oracle_population": list(CONFORMANCE_ORACLE_POPULATION),
        "excluded_semantic_mismatch_population": list(EXCLUDED_SEMANTIC_MISMATCH_POPULATION),
        "mitigation_population": list(MITIGATION_POPULATION),
        "reserve_detector_population": list(RESERVE_DETECTOR_POPULATION),
        "comparators": normalized,
        "status": "CENSUS_VALID",
        "league_status": "NOT_SCOREABLE",
        "scoreable": False,
        "scoreable_blockers": [
            "natural_public_case_ledger_missing",
            "independent_gold_receipts_missing",
            "same_input_decode_parity_receipts_missing",
            "equal_budget_three_repeat_receipts_missing",
        ],
        "external_claim_authorized": False,
    }
    _CENSUS_VALIDATION_CACHE[cache_key] = dict(result)
    return dict(result)


def write_comparator_census_receipt(
    manifest_path: Path | str,
    artifact_root: Path | str,
    receipt_path: Path | str,
) -> dict[str, object]:
    """Reopen a census manifest and write its validated, hash-bound receipt."""
    source = Path(manifest_path).resolve()
    destination = Path(receipt_path).resolve()
    if destination.exists():
        raise FileExistsError(f"comparator census receipt already exists: {destination}")
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("comparator census manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise ExternalLeagueError("comparator census manifest root is invalid")
    receipt = validate_comparator_census(manifest, artifact_root=artifact_root)
    receipt = {
        **receipt,
        "source_manifest": {
            "path": str(source),
            "sha256": _sha256_file(source),
        },
    }
    _write_json(destination, receipt)
    return {**receipt, "receipt": str(destination)}


def _load_execution_census_entry(
    census_receipt: Path | str,
    artifact_root: Path | str,
    comparator_name: str,
) -> tuple[dict[str, object], Path]:
    receipt_path = Path(census_receipt).resolve()
    try:
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("comparator census receipt is unreadable") from exc
    if not isinstance(stored, Mapping) or stored.get("schema") != COMPARATOR_CENSUS_RECEIPT_SCHEMA:
        raise ExternalLeagueError("comparator census receipt schema is invalid")
    source = stored.get("source_manifest")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
        raise ExternalLeagueError("comparator census receipt does not bind its source manifest")
    source_path = Path(str(source["path"])).resolve()
    if not source_path.is_file() or _sha256_file(source_path) != source["sha256"]:
        raise ExternalLeagueError("comparator census source manifest hash mismatches")
    try:
        manifest = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("comparator census source manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise ExternalLeagueError("comparator census source manifest root is invalid")
    validated = validate_comparator_census(manifest, artifact_root=artifact_root)
    for field in (
        "manifest_sha256",
        "artifact_root",
        "detector_population",
        "conformance_oracle_population",
        "excluded_semantic_mismatch_population",
        "mitigation_population",
        "reserve_detector_population",
        "comparators",
        "status",
        "league_status",
        "scoreable",
        "scoreable_blockers",
        "external_claim_authorized",
    ):
        if stored.get(field) != validated[field]:
            raise ExternalLeagueError(f"comparator census receipt was altered after validation: {field}")
    entry = next(
        (item for item in validated["comparators"] if item["name"] == comparator_name),
        None,
    )
    if entry is None:
        raise ExternalLeagueError(f"comparator is absent from the frozen census: {comparator_name}")
    return dict(entry), receipt_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hashes(frames: np.ndarray) -> list[str]:
    return [_sha256_bytes(frame.tobytes()) for frame in frames]


def pack_renderer_png_sequence(
    frame_directory: Path | str,
    output_root: Path | str,
    *,
    fps: int,
) -> dict[str, object]:
    """Freeze an actual Godot PNG capture sequence as a detector NPZ input.

    This is deliberately stricter than a screenshot importer.  Each source
    image must use one fixed zero-padded ``frame_<tick>.png`` convention
    through a contiguous final tick,
    have identical dimensions, and decode as RGB pixels.  The receipt retains
    every source PNG hash, derives the presentation timestamps from the
    declared fixed physics cadence, and binds the packed RGB bytes.  It makes
    a public project's renderer output usable by the same CFR lane as the
    external comparators without pretending that sparse screenshots are video.
    """
    source = Path(frame_directory).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"renderer PNG pack output already exists: {root}")
    if not source.is_dir():
        raise ExternalLeagueError("renderer PNG capture directory is missing")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ExternalLeagueError("renderer PNG capture requires a positive fixed FPS")
    discovered = list(source.glob("frame_*.png"))
    if not discovered:
        raise ExternalLeagueError("renderer PNG capture contains no frames")
    numbered: list[tuple[int, int, Path]] = []
    for path in discovered:
        match = re.fullmatch(r"frame_(0[0-9]+)\.png", path.name)
        if match is None:
            raise ExternalLeagueError("renderer PNG capture frame names must be zero-padded ticks")
        numbered.append((int(match.group(1)), len(match.group(1)), path))
    padding = {width for _, width, _ in numbered}
    if len(padding) != 1:
        raise ExternalLeagueError("renderer PNG capture frame names must use one fixed tick padding width")
    numbered.sort()
    if [index for index, _, _ in numbered] != list(range(len(numbered))):
        raise ExternalLeagueError("renderer PNG capture frame names must be contiguous from tick zero")
    paths = [path for _, _, path in numbered]
    frames: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for path in paths:
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ExternalLeagueError(f"renderer PNG frame cannot be decoded: {path.name}")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if shape is None:
            shape = rgb.shape
        elif rgb.shape != shape:
            raise ExternalLeagueError("renderer PNG capture dimensions differ between ticks")
        frames.append(rgb)
    frame_array = np.stack(frames)
    if frame_array.dtype != np.uint8 or frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ExternalLeagueError("renderer PNG capture does not decode to uint8 RGB")
    timestamps = np.arange(len(frame_array), dtype=np.float64) / float(fps)
    root.mkdir(parents=True)
    artifact = root / "renderer-frames.npz"
    np.savez_compressed(artifact, frames=frame_array, timestamps=timestamps)
    receipt = {
        "schema": "flashpatch-external-godot-renderer-pack-v1",
        "source_capture": {
            "directory": str(source),
            "frame_count": len(paths),
            "frames": [
                {"path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
                for path in paths
            ],
        },
        "cfr": {
            "fps": fps,
            "frame_count": len(frame_array),
            "timestamps_us": (timestamps * 1_000_000).round().astype(np.int64).tolist(),
        },
        "renderer_rgb": {
            "raw_sha256": _sha256_bytes(frame_array.tobytes()),
            "shape": list(frame_array.shape),
            "frame_sha256": _frame_hashes(frame_array),
        },
        "artifact": {"path": artifact.name, "sha256": _sha256_file(artifact), "bytes": artifact.stat().st_size},
        "status": "RENDERER_SEQUENCE_PACKED",
        "scoreable": False,
        "scoreable_blockers": [
            "source_execution_receipt_missing",
            "independent_gold_receipt_missing",
            "frozen_public_case_ledger_missing",
        ],
    }
    receipt_path = root / "renderer-pack-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_tree_manifest(root: Path) -> list[dict[str, object]]:
    """Hash a dependency tree without following links outside its root."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ExternalLeagueError("EA IRIS dependency root is unavailable")
    rows: list[dict[str, object]] = []
    for path in sorted(resolved_root.rglob("*"), key=lambda item: item.relative_to(resolved_root).as_posix()):
        relative = path.relative_to(resolved_root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise ExternalLeagueError("EA IRIS dependency root contains an absolute symlink")
            try:
                path.resolve(strict=False).relative_to(resolved_root)
            except ValueError as exc:
                raise ExternalLeagueError("EA IRIS dependency symlink escapes its frozen root") from exc
            rows.append({"path": relative, "type": "symlink", "target": target})
        elif path.is_file():
            rows.append({
                "path": relative,
                "type": "file",
                "bytes": path.stat().st_size,
                "mode": path.stat().st_mode & 0o777,
                "sha256": _sha256_file(path),
            })
        elif path.is_dir():
            continue
        else:
            raise ExternalLeagueError("EA IRIS dependency root contains a non-file artifact")
    if not rows:
        raise ExternalLeagueError("EA IRIS dependency root is empty")
    return rows


def _parse_cmake_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ExternalLeagueError("EA IRIS minimal OpenCV CMake cache is missing")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw_line or raw_line.startswith(("#", "//")) or "=" not in raw_line:
            continue
        identity, value = raw_line.split("=", 1)
        name = identity.split(":", 1)[0]
        if not name or name in values:
            raise ExternalLeagueError("EA IRIS minimal OpenCV CMake cache is ambiguous")
        values[name] = value
    return values


def _read_debian_archive_identity(archive: Path, *, dpkg_deb: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(dpkg_deb), "-f", str(archive), "Package", "Version", "Architecture"],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    if completed.returncode != 0 or set(fields) != {"Package", "Version", "Architecture"}:
        raise ExternalLeagueError("EA IRIS minimal support archive metadata is invalid")
    return {
        "package": fields["Package"],
        "version": fields["Version"],
        "architecture": fields["Architecture"],
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": _sha256_file(archive),
    }


def _copy_relative_symlink_tree(source: Path, destination: Path) -> None:
    """Copy a frozen tree while retaining only root-contained relative links."""
    source_root = source.resolve()
    if not source_root.is_dir():
        raise ExternalLeagueError("EA IRIS minimal media source tree is missing")
    for path in sorted(source_root.rglob("*"), key=lambda item: item.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root)
        target = destination / relative
        if path.is_symlink():
            link = os.readlink(path)
            if Path(link).is_absolute():
                raise ExternalLeagueError("EA IRIS minimal media source has an absolute symlink")
            try:
                path.resolve(strict=False).relative_to(source_root)
            except ValueError as exc:
                raise ExternalLeagueError("EA IRIS minimal media symlink escapes source root") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link)
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise ExternalLeagueError("EA IRIS minimal media source has a non-file artifact")


def _readelf_dynamic_contract(binary: Path, *, readelf: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(readelf), "-d", str(binary)],
        capture_output=True,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    if completed.returncode != 0:
        raise ExternalLeagueError("EA IRIS minimal media ELF cannot be inspected")
    text = completed.stdout.decode("utf-8", "replace")
    needed = sorted(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text))
    search_paths = sorted(re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]*)\]", text))
    forbidden_path_fragments = ("/tmp/", "/home/", "/usr/local/", "/opt/")
    if any(fragment in value for value in search_paths for fragment in forbidden_path_fragments):
        raise ExternalLeagueError("EA IRIS minimal media ELF contains a host-specific runtime path")
    if any(value and value != "$ORIGIN" for value in search_paths):
        raise ExternalLeagueError("EA IRIS minimal media library runtime path is not empty or $ORIGIN")
    return {
        "needed": needed,
        "search_paths": search_paths,
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stderr_sha256": _sha256_bytes(completed.stderr),
    }


def _audit_iris_minimal_media_sdk(sdk_root: Path, *, readelf: Path) -> dict[str, object]:
    root = sdk_root.resolve()
    library_root = root / "lib"
    expected_real_libraries = {
        "libopencv_core.so.4.8.0",
        "libopencv_imgproc.so.4.8.0",
        "libopencv_features2d.so.4.8.0",
        "libopencv_imgcodecs.so.4.8.0",
        "libopencv_videoio.so.4.8.0",
        "libavcodec.so.60.31.102",
        "libavformat.so.60.16.100",
        "libavutil.so.58.29.100",
        "libswscale.so.7.5.100",
        "libspdlog.so.1.12.0",
        "libfmt.so.9.1.0",
        "libcrypto.so.3",
    }
    real_libraries = {
        path.name for path in library_root.iterdir()
        if path.is_file() and not path.is_symlink() and ".so" in path.name
    }
    if real_libraries != expected_real_libraries:
        raise ExternalLeagueError("EA IRIS minimal media SDK real library set is not exact")
    forbidden_needed_fragments = (
        "gstreamer", "gobject", "glib-", "gdal", "gdcm", "openexr", "imath",
        "gtk", "qt", "x11", "glx", "opengl", "tbb", "lapack", "blas",
        "gfortran", "jpeg", "png", "tiff", "webp", "openjp", "v4l",
    )
    rows: dict[str, dict[str, object]] = {}
    for name in sorted(expected_real_libraries):
        path = library_root / name
        dynamic = _readelf_dynamic_contract(path, readelf=readelf)
        if any(fragment in needed.lower() for needed in dynamic["needed"] for fragment in forbidden_needed_fragments):
            raise ExternalLeagueError("EA IRIS minimal media SDK retained a forbidden dynamic dependency")
        rows[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "dynamic": dynamic,
        }
    required_headers = (
        "include/opencv4/opencv2/core.hpp",
        "include/opencv4/opencv2/imgproc.hpp",
        "include/opencv4/opencv2/features2d.hpp",
        "include/opencv4/opencv2/videoio.hpp",
        "include/libavformat/avformat.h",
        "include/libavcodec/avcodec.h",
        "include/libavutil/avutil.h",
        "include/libswscale/swscale.h",
        "include/nlohmann/json.hpp",
        "include/spdlog/spdlog.h",
        "include/fmt/format.h",
        "include/openssl/sha.h",
        "include/x86_64-linux-gnu/openssl/opensslconf.h",
    )
    if any(not (root / relative).is_file() for relative in required_headers):
        raise ExternalLeagueError("EA IRIS minimal media SDK required header set is incomplete")
    return {
        "real_libraries": rows,
        "real_libraries_sha256": _canonical_json_sha256(rows),
        "required_headers": list(required_headers),
        "forbidden_dynamic_dependencies_absent": True,
        "runtime_path_policy": "EMPTY_OR_ORIGIN_ONLY_V1",
    }


def freeze_iris_minimal_media_toolchain(spec: IrisMinimalMediaBuildSpec) -> dict[str, object]:
    """Materialize and freeze the exact source-built media SDK used by d969 IRIS."""
    sdk_root = spec.sdk_root.resolve()
    if sdk_root.exists():
        raise FileExistsError(f"EA IRIS minimal media SDK already exists: {sdk_root}")
    archives = {
        "opencv": spec.opencv_archive.resolve(),
        "ffmpeg": spec.ffmpeg_archive.resolve(),
    }
    expected_sources = {
        "opencv": EA_IRIS_MINIMAL_OPENCV_SOURCE,
        "ffmpeg": EA_IRIS_MINIMAL_FFMPEG_SOURCE,
    }
    for name, archive in archives.items():
        if not archive.is_file() or _sha256_file(archive) != expected_sources[name]["archive_sha256"]:
            raise ExternalLeagueError(f"EA IRIS pinned {name} source archive drifted")
    opencv_source = spec.opencv_source.resolve()
    ffmpeg_source = spec.ffmpeg_source.resolve()
    opencv_license = opencv_source / "LICENSE"
    ffmpeg_license = ffmpeg_source / "COPYING.LGPLv2.1"
    if (
        not opencv_license.is_file()
        or _sha256_file(opencv_license) != EA_IRIS_MINIMAL_OPENCV_SOURCE["license_sha256"]
        or not ffmpeg_license.is_file()
        or _sha256_file(ffmpeg_license) != EA_IRIS_MINIMAL_FFMPEG_SOURCE["license_sha256"]
    ):
        raise ExternalLeagueError("EA IRIS minimal media source license evidence drifted")

    cache_path = spec.opencv_build.resolve() / "CMakeCache.txt"
    cache = _parse_cmake_cache(cache_path)
    observed_modules = tuple(cache.get("OPENCV_MODULES_BUILD", "").split(";"))
    if observed_modules != EA_IRIS_MINIMAL_OPENCV_MODULES:
        raise ExternalLeagueError("EA IRIS minimal OpenCV effective module closure drifted")
    if cache.get("WITH_FFMPEG") != "ON" or cache.get("VIDEOIO_ENABLE_PLUGINS") != "OFF":
        raise ExternalLeagueError("EA IRIS minimal OpenCV FFmpeg backend contract drifted")
    for key, value in EA_IRIS_MINIMAL_OPENCV_FORBIDDEN_CACHE.items():
        if cache.get(key) != value:
            raise ExternalLeagueError(f"EA IRIS minimal OpenCV forbidden feature was not disabled: {key}")
    cmake_command = list(spec.opencv_cmake_command)
    if (
        not cmake_command
        or "-DBUILD_LIST=core,imgproc,features2d,videoio" not in cmake_command
        or "-DWITH_FFMPEG=ON" not in cmake_command
        or "-DVIDEOIO_ENABLE_PLUGINS=OFF" not in cmake_command
    ):
        raise ExternalLeagueError("EA IRIS minimal OpenCV command omits its exact module/backend boundary")

    ffmpeg_config = ffmpeg_source / "ffbuild" / "config.mak"
    ffmpeg_header = ffmpeg_source / "config.h"
    if not ffmpeg_config.is_file() or not ffmpeg_header.is_file():
        raise ExternalLeagueError("EA IRIS minimal FFmpeg configured evidence is missing")
    config_text = ffmpeg_config.read_text(encoding="utf-8", errors="strict")
    header_text = ffmpeg_header.read_text(encoding="utf-8", errors="strict")
    for row in ("CONFIG_FFV1_DECODER=yes", "CONFIG_MATROSKA_DEMUXER=yes", "CONFIG_FILE_PROTOCOL=yes"):
        if row not in config_text:
            raise ExternalLeagueError("EA IRIS minimal FFmpeg enabled component closure drifted")
    for row in ("#define CONFIG_AVDEVICE 0", "#define CONFIG_AVFILTER 0", "#define CONFIG_SWRESAMPLE 0"):
        if row not in header_text:
            raise ExternalLeagueError("EA IRIS minimal FFmpeg forbidden library closure drifted")
    ffmpeg_command = list(spec.ffmpeg_configure_command)
    required_ffmpeg_flags = {
        "--disable-autodetect", "--disable-everything", "--disable-static", "--enable-shared",
        "--disable-programs", "--disable-network", "--disable-avdevice", "--disable-avfilter",
        "--disable-swresample", "--enable-avcodec", "--enable-avformat", "--enable-avutil",
        "--enable-swscale", "--enable-decoder=ffv1", "--enable-demuxer=matroska",
        "--enable-protocol=file",
    }
    if not required_ffmpeg_flags <= set(ffmpeg_command):
        raise ExternalLeagueError("EA IRIS minimal FFmpeg configure command is not fail-closed")

    dpkg_deb = spec.dpkg_deb.resolve()
    support_rows = [
        _read_debian_archive_identity(path.resolve(), dpkg_deb=dpkg_deb)
        for path in spec.support_archives
    ]
    observed_support = {
        str(row["package"]): (str(row["version"]), str(row["architecture"]), str(row["sha256"]))
        for row in support_rows
    }
    if observed_support != EA_IRIS_MINIMAL_SUPPORT_CLOSURE:
        raise ExternalLeagueError("EA IRIS minimal support archive closure drifted")
    build_tool_rows = [
        _read_debian_archive_identity(path.resolve(), dpkg_deb=dpkg_deb)
        for path in spec.build_tool_archives
    ]
    observed_build_tools = {
        str(row["package"]): (str(row["version"]), str(row["architecture"]), str(row["sha256"]))
        for row in build_tool_rows
    }
    if observed_build_tools != EA_IRIS_MINIMAL_BUILD_TOOL_CLOSURE:
        raise ExternalLeagueError("EA IRIS minimal build-tool archive closure drifted")

    sdk_root.mkdir(parents=True)
    include_root = sdk_root / "include"
    library_root = sdk_root / "lib"
    license_root = sdk_root / "licenses"
    evidence_root = sdk_root / "build-evidence"
    for path in (include_root, library_root, license_root, evidence_root):
        path.mkdir()
    _copy_relative_symlink_tree(spec.opencv_install.resolve() / "include" / "opencv4", include_root / "opencv4")
    for relative in ("libavformat", "libavcodec", "libavutil", "libswscale"):
        _copy_relative_symlink_tree(spec.ffmpeg_install.resolve() / "include" / relative, include_root / relative)
    support_root = spec.support_root.resolve()
    for relative in ("nlohmann", "spdlog", "fmt", "openssl"):
        _copy_relative_symlink_tree(support_root / "usr" / "include" / relative, include_root / relative)
    _copy_relative_symlink_tree(
        support_root / "usr" / "include" / "x86_64-linux-gnu" / "openssl",
        include_root / "x86_64-linux-gnu" / "openssl",
    )
    library_sources = (
        (spec.opencv_install.resolve() / "lib", ("libopencv_core.so*", "libopencv_imgproc.so*", "libopencv_features2d.so*", "libopencv_imgcodecs.so*", "libopencv_videoio.so*")),
        (spec.ffmpeg_install.resolve() / "lib", ("libavcodec.so*", "libavformat.so*", "libavutil.so*", "libswscale.so*")),
        (support_root / "usr" / "lib" / "x86_64-linux-gnu", ("libspdlog.so*", "libfmt.so*", "libcrypto.so*")),
    )
    copied_names: set[str] = set()
    for source_root, patterns in library_sources:
        for pattern in patterns:
            for source_path in sorted(source_root.glob(pattern)):
                if source_path.name in copied_names:
                    continue
                copied_names.add(source_path.name)
                target = library_root / source_path.name
                if source_path.is_symlink():
                    link = os.readlink(source_path)
                    if Path(link).is_absolute():
                        raise ExternalLeagueError("EA IRIS minimal media library has an absolute symlink")
                    target.symlink_to(link)
                elif source_path.is_file():
                    shutil.copy2(source_path, target)
    shutil.copy2(opencv_license, license_root / "OpenCV-Apache-2.0.txt")
    shutil.copy2(ffmpeg_license, license_root / "FFmpeg-LGPL-2.1-or-later.txt")
    shutil.copy2(cache_path, evidence_root / "OpenCV-CMakeCache.txt")
    shutil.copy2(spec.opencv_build.resolve() / "CMakeVars.txt", evidence_root / "OpenCV-CMakeVars.txt")
    shutil.copy2(ffmpeg_config, evidence_root / "FFmpeg-config.mak")
    shutil.copy2(ffmpeg_header, evidence_root / "FFmpeg-config.h")
    ffmpeg_pkgconfig_root = spec.ffmpeg_install.resolve() / "lib" / "pkgconfig"
    _copy_relative_symlink_tree(ffmpeg_pkgconfig_root, evidence_root / "FFmpeg-pkgconfig")
    pkgconfig_rows = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted((evidence_root / "FFmpeg-pkgconfig").glob("*.pc"))
    ]
    if [row["name"] for row in pkgconfig_rows] != [
        "libavcodec.pc", "libavformat.pc", "libavutil.pc", "libswscale.pc",
    ]:
        raise ExternalLeagueError("EA IRIS minimal FFmpeg pkg-config metadata set is not exact")
    static_archives = sorted(path.name for path in spec.ffmpeg_install.resolve().glob("lib/*.a"))
    if static_archives:
        raise ExternalLeagueError("EA IRIS minimal FFmpeg install unexpectedly contains static archives")
    audit = _audit_iris_minimal_media_sdk(sdk_root, readelf=spec.readelf.resolve())
    sdk_tree = _path_tree_manifest(sdk_root)
    receipt = {
        "schema": EA_IRIS_MINIMAL_MEDIA_SCHEMA,
        "sources": {
            "opencv": {**EA_IRIS_MINIMAL_OPENCV_SOURCE, "archive": str(archives["opencv"])},
            "ffmpeg": {**EA_IRIS_MINIMAL_FFMPEG_SOURCE, "archive": str(archives["ffmpeg"])},
        },
        "source_required_opencv_modules": EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS,
        "opencv": {
            "effective_modules": list(observed_modules),
            "forbidden_cache": EA_IRIS_MINIMAL_OPENCV_FORBIDDEN_CACHE,
            "cmake_command": cmake_command,
            "cache_sha256": _sha256_file(cache_path),
            "cache_copy_sha256": _sha256_file(evidence_root / "OpenCV-CMakeCache.txt"),
            "cmake_vars_sha256": _sha256_file(spec.opencv_build.resolve() / "CMakeVars.txt"),
        },
        "ffmpeg": {
            "enabled_components": {key: list(value) for key, value in EA_IRIS_MINIMAL_FFMPEG_COMPONENTS.items()},
            "configure_command": ffmpeg_command,
            "config_mak_sha256": _sha256_file(ffmpeg_config),
            "config_h_sha256": _sha256_file(ffmpeg_header),
            "pkgconfig": pkgconfig_rows,
            "pkgconfig_sha256": _canonical_json_sha256(pkgconfig_rows),
            "static_archives": static_archives,
        },
        "support_archives": sorted(support_rows, key=lambda row: str(row["package"])),
        "support_archives_sha256": _canonical_json_sha256(sorted(support_rows, key=lambda row: str(row["package"]))),
        "build_tool_archives": sorted(build_tool_rows, key=lambda row: str(row["package"])),
        "build_tool_archives_sha256": _canonical_json_sha256(sorted(build_tool_rows, key=lambda row: str(row["package"]))),
        "build_tool_closure": {
            "declared_archives_exact": True,
            "host_coreutils_archive": "NOT_VERIFIED_CURRENT_ARCHIVE_REVISION_DRIFT",
            "independent_machine_rebuild": "NOT_CLAIMED",
        },
        "sdk": {
            "root": str(sdk_root),
            "tree": sdk_tree,
            "tree_sha256": _canonical_json_sha256(sdk_tree),
            "elf_audit": audit,
        },
        "status": "MINIMAL_MEDIA_BUILD_VERIFIED",
        "scoreable": False,
        "scoreable_blockers": [
            "iris_build_not_yet_verified",
            "conformance_not_yet_verified",
            "host_coreutils_archive_not_closed",
        ],
    }
    receipt_path = sdk_root.parent / f"{sdk_root.name}-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"EA IRIS minimal media receipt already exists: {receipt_path}")
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": _sha256_file(receipt_path)}


def _load_iris_minimal_media_toolchain(receipt_ref: Path | str) -> tuple[dict[str, object], Path]:
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS minimal media receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EA_IRIS_MINIMAL_MEDIA_SCHEMA
        or payload.get("status") != "MINIMAL_MEDIA_BUILD_VERIFIED"
        or payload.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS minimal media receipt state is invalid")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ExternalLeagueError("EA IRIS minimal media source ledger is missing")
    for name, expected in (("opencv", EA_IRIS_MINIMAL_OPENCV_SOURCE), ("ffmpeg", EA_IRIS_MINIMAL_FFMPEG_SOURCE)):
        row = sources.get(name)
        if not isinstance(row, Mapping) or not isinstance(row.get("archive"), str):
            raise ExternalLeagueError("EA IRIS minimal media source archive ledger is invalid")
        archive = Path(str(row["archive"])).resolve()
        expected_row = {**expected, "archive": str(archive)}
        if dict(row) != expected_row or not archive.is_file() or _sha256_file(archive) != expected["archive_sha256"]:
            raise ExternalLeagueError("EA IRIS minimal media source archive drifted after freeze")
    sdk = payload.get("sdk")
    if not isinstance(sdk, Mapping) or not isinstance(sdk.get("root"), str):
        raise ExternalLeagueError("EA IRIS minimal media SDK ledger is missing")
    sdk_root = Path(str(sdk["root"])).resolve()
    fresh_tree = _path_tree_manifest(sdk_root)
    readelf = Path(EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN["readelf"][0]).resolve()
    fresh_audit = _audit_iris_minimal_media_sdk(sdk_root, readelf=readelf)
    if (
        sdk.get("tree") != fresh_tree
        or sdk.get("tree_sha256") != _canonical_json_sha256(fresh_tree)
        or sdk.get("elf_audit") != fresh_audit
    ):
        raise ExternalLeagueError("EA IRIS minimal media SDK drifted after freeze")
    if payload.get("source_required_opencv_modules") != EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS:
        raise ExternalLeagueError("EA IRIS minimal media source-module provenance drifted")
    opencv = payload.get("opencv")
    ffmpeg = payload.get("ffmpeg")
    if (
        not isinstance(opencv, Mapping)
        or opencv.get("effective_modules") != list(EA_IRIS_MINIMAL_OPENCV_MODULES)
        or opencv.get("forbidden_cache") != EA_IRIS_MINIMAL_OPENCV_FORBIDDEN_CACHE
        or not isinstance(ffmpeg, Mapping)
        or ffmpeg.get("enabled_components") != {key: list(value) for key, value in EA_IRIS_MINIMAL_FFMPEG_COMPONENTS.items()}
    ):
        raise ExternalLeagueError("EA IRIS minimal media configured contract drifted")
    pkgconfig_rows = ffmpeg.get("pkgconfig")
    if (
        not isinstance(pkgconfig_rows, list)
        or [row.get("name") for row in pkgconfig_rows if isinstance(row, Mapping)]
        != ["libavcodec.pc", "libavformat.pc", "libavutil.pc", "libswscale.pc"]
        or ffmpeg.get("pkgconfig_sha256") != _canonical_json_sha256(pkgconfig_rows)
        or ffmpeg.get("static_archives") != []
    ):
        raise ExternalLeagueError("EA IRIS minimal FFmpeg linker metadata drifted")
    for row in pkgconfig_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ExternalLeagueError("EA IRIS minimal FFmpeg pkg-config row is invalid")
        artifact = sdk_root / "build-evidence" / "FFmpeg-pkgconfig" / str(row["name"])
        if (
            not artifact.is_file()
            or row.get("bytes") != artifact.stat().st_size
            or row.get("sha256") != _sha256_file(artifact)
        ):
            raise ExternalLeagueError("EA IRIS minimal FFmpeg pkg-config bytes drifted")
    support_rows = payload.get("support_archives")
    if not isinstance(support_rows, list) or payload.get("support_archives_sha256") != _canonical_json_sha256(support_rows):
        raise ExternalLeagueError("EA IRIS minimal media support ledger hash drifted")
    observed_support = {
        str(row.get("package")): (str(row.get("version")), str(row.get("architecture")), str(row.get("sha256")))
        for row in support_rows if isinstance(row, Mapping)
    }
    if observed_support != EA_IRIS_MINIMAL_SUPPORT_CLOSURE:
        raise ExternalLeagueError("EA IRIS minimal media support archive closure drifted")
    for row in support_rows:
        archive = Path(str(row.get("path", ""))).resolve()
        if not archive.is_file() or row.get("bytes") != archive.stat().st_size or row.get("sha256") != _sha256_file(archive):
            raise ExternalLeagueError("EA IRIS minimal media support archive bytes drifted")
    build_tool_rows = payload.get("build_tool_archives")
    if (
        not isinstance(build_tool_rows, list)
        or payload.get("build_tool_archives_sha256") != _canonical_json_sha256(build_tool_rows)
        or payload.get("build_tool_closure") != {
            "declared_archives_exact": True,
            "host_coreutils_archive": "NOT_VERIFIED_CURRENT_ARCHIVE_REVISION_DRIFT",
            "independent_machine_rebuild": "NOT_CLAIMED",
        }
    ):
        raise ExternalLeagueError("EA IRIS minimal build-tool archive ledger drifted")
    observed_build_tools = {
        str(row.get("package")): (str(row.get("version")), str(row.get("architecture")), str(row.get("sha256")))
        for row in build_tool_rows if isinstance(row, Mapping)
    }
    if observed_build_tools != EA_IRIS_MINIMAL_BUILD_TOOL_CLOSURE:
        raise ExternalLeagueError("EA IRIS minimal build-tool archive closure drifted")
    for row in build_tool_rows:
        archive = Path(str(row.get("path", ""))).resolve()
        if not archive.is_file() or row.get("bytes") != archive.stat().st_size or row.get("sha256") != _sha256_file(archive):
            raise ExternalLeagueError("EA IRIS minimal build-tool archive bytes drifted")
    return dict(payload), receipt_path


def _iris_source_checkout_receipt(checkout: Path) -> dict[str, object]:
    source = checkout.resolve()
    if not source.is_dir():
        raise ExternalLeagueError("EA IRIS source checkout is unavailable")
    trusted_git_path, trusted_git_sha256 = EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN["git"]
    git_binary = Path(trusted_git_path).resolve()
    if not git_binary.is_file() or _sha256_file(git_binary) != trusted_git_sha256:
        raise ExternalLeagueError("EA IRIS source checkout trusted git anchor is unavailable")

    def git(*args: str) -> str:
        completed = subprocess.run(
            [str(git_binary), "-C", str(source), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        if completed.returncode != 0:
            raise ExternalLeagueError("EA IRIS source checkout git provenance cannot be read")
        return completed.stdout.strip()

    revision = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    origin = git("config", "--get", "remote.origin.url")
    if (
        revision != EA_IRIS_SOURCE_REVISION
        or tree != EA_IRIS_SOURCE_TREE
        or status
        or _normalized_repository_url(origin) != "https://github.com/electronicarts/IRIS"
    ):
        raise ExternalLeagueError("EA IRIS source checkout is not the clean frozen d96978ac tree")
    tracked = git("ls-files", "-z")
    tracked_paths = [part for part in tracked.split("\0") if part]
    if not tracked_paths:
        raise ExternalLeagueError("EA IRIS source checkout has no tracked files")
    source_rows = []
    for relative in sorted(tracked_paths):
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise ExternalLeagueError("EA IRIS tracked source path escapes checkout") from exc
        if not path.is_file():
            raise ExternalLeagueError("EA IRIS tracked source artifact is missing")
        source_rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    required_hashes = {
        "LICENSE.txt": EA_IRIS_SOURCE_LICENSE_SHA256,
        "config/appsettings.json": EA_IRIS_SOURCE_CONFIG_SHA256,
        "example/main.cpp": EA_IRIS_SOURCE_EXAMPLE_SHA256,
        "vcpkg.json": EA_IRIS_SOURCE_VCPKG_MANIFEST_SHA256,
    }
    observed = {row["path"]: row["sha256"] for row in source_rows}
    if any(observed.get(path) != digest for path, digest in required_hashes.items()):
        raise ExternalLeagueError("EA IRIS source checkout required artifact hash drifted")
    if not all((source / relative).is_file() for relative in EA_IRIS_SOURCE_CPP_PATHS):
        raise ExternalLeagueError("EA IRIS source checkout omits a compiled upstream source")
    license_text = (source / "LICENSE.txt").read_text(encoding="utf-8", errors="replace")
    if "Redistribution and use in source and binary forms" not in license_text:
        raise ExternalLeagueError("EA IRIS source checkout license is not BSD-3-Clause evidence")
    return {
        "repository_url": "https://github.com/electronicarts/IRIS",
        "revision": revision,
        "tree": tree,
        "clean": True,
        "path": str(source),
        "tracked_files": source_rows,
        "tracked_files_sha256": _canonical_json_sha256(source_rows),
    }


def _iris_trusted_build_tools(spec: IrisSourceBuildSpec) -> dict[str, Path]:
    tools = {
        "compiler": spec.compiler.resolve(),
        "archiver": spec.archiver.resolve(),
        "dpkg_deb": spec.dpkg_deb.resolve(),
        "readelf": spec.readelf.resolve(),
        "ldd": spec.ldd.resolve(),
    }
    for name, tool in tools.items():
        expected_path, expected_sha256 = EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN[name]
        if (
            not tool.is_file()
            or not os.access(tool, os.X_OK)
            or tool != Path(expected_path).resolve()
            or _sha256_file(tool) != expected_sha256
        ):
            raise ExternalLeagueError(f"EA IRIS source build trusted toolchain anchor drifted: {name}")
    return tools


def _freeze_iris_source_build_manifest_minimal(
    spec: IrisSourceBuildSpec,
    output: Path,
) -> dict[str, object]:
    if spec.minimal_media_receipt is None:
        raise ExternalLeagueError("EA IRIS minimal media receipt is required")
    source = _iris_source_checkout_receipt(spec.source_checkout)
    source_root = spec.source_checkout.resolve()
    pattern_source = source_root / "src" / "PatternDetection.cpp"
    pattern_requirement = EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS["src/PatternDetection.cpp"]
    video_source = source_root / "src" / "VideoAnalyser.cpp"
    video_requirement = EA_IRIS_MINIMAL_FFMPEG_SOURCE_REQUIREMENTS["src/VideoAnalyser.cpp"]
    video_text = video_source.read_text(encoding="utf-8")
    example_source = source_root / "example" / "main.cpp"
    if (
        _sha256_file(pattern_source) != pattern_requirement["sha256"]
        or pattern_requirement["required_header"] not in pattern_source.read_text(encoding="utf-8")
        or _sha256_file(video_source) != video_requirement["sha256"]
        or any(header not in video_text for header in video_requirement["required_headers"])
        or any(call not in video_text for call in video_requirement["required_calls"])
        or _sha256_file(example_source) != EA_IRIS_SOURCE_EXAMPLE_SHA256
        or "vA.AnalyseVideo" not in example_source.read_text(encoding="utf-8")
    ):
        raise ExternalLeagueError("EA IRIS source-required OpenCV module provenance drifted")
    media, media_receipt_path = _load_iris_minimal_media_toolchain(spec.minimal_media_receipt)
    dependency_root = Path(str(media["sdk"]["root"])).resolve()
    if spec.dependency_root.resolve() != dependency_root:
        raise ExternalLeagueError("EA IRIS source dependency root differs from minimal media SDK")
    tools = _iris_trusted_build_tools(spec)
    compiler_version = subprocess.run(
        [str(tools["compiler"]), "--version"],
        capture_output=True,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    if compiler_version.returncode != 0:
        raise ExternalLeagueError("EA IRIS compiler version cannot be observed")
    build_environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "SOURCE_DATE_EPOCH": "1710356805",
        "ZERO_AR_DATE": "1",
    }
    tool_rows = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in tools.items()
    }
    adapter_hash = _sha256_bytes((_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n").encode("utf-8"))
    manifest = {
        "schema": "flashpatch-l7-ea-iris-source-build-manifest-v2",
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "distribution": "local-source-build-d96978ac-not-official-release",
        "source": source,
        "adapter": {
            "source_sha256": adapter_hash,
            "public_api_sequence": list(EA_IRIS_SOURCE_BOUNDARY_METHODS),
            "input_contract": "canonical-60fps-cfr-ffv1-rgb24-v1",
        },
        "dependency_closure": {
            "kind": "PINNED_MINIMAL_OPENCV_4_8_0_FFMPEG_N6_1_1_SOURCE_BUILD_V1",
            "root": str(dependency_root),
            "tree_sha256": media["sdk"]["tree_sha256"],
            "minimal_media_receipt": {
                "path": str(media_receipt_path),
                "sha256": _sha256_file(media_receipt_path),
            },
            "effective_opencv_modules": list(EA_IRIS_MINIMAL_OPENCV_MODULES),
            "build_tool_closure": media["build_tool_closure"],
            "source_module_provenance": {
                "opencv": EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS,
                "ffmpeg": EA_IRIS_MINIMAL_FFMPEG_SOURCE_REQUIREMENTS,
            },
        },
        "toolchain": {
            **tool_rows,
            "compiler_version_stdout_sha256": _sha256_bytes(compiler_version.stdout),
            "compiler_version_stderr_sha256": _sha256_bytes(compiler_version.stderr),
        },
        "build_environment": build_environment,
        "build_environment_sha256": _canonical_json_sha256(build_environment),
        "scoreable": False,
        "scoreable_blockers": [
            "build_not_yet_executed",
            "source_video_conformance_not_yet_verified",
            "temporal_boundary_conformance_not_yet_verified",
            "local_execution_witness_not_independent",
            "minimal_media_host_coreutils_archive_not_closed",
        ],
    }
    _write_json(output, manifest)
    return {**manifest, "manifest": str(output), "manifest_sha256": _sha256_file(output)}


def freeze_iris_source_build_manifest(
    spec: IrisSourceBuildSpec,
    destination: Path | str,
) -> dict[str, object]:
    """Freeze source, compiler and extracted Debian dependency bytes before build."""
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"EA IRIS source build manifest already exists: {output}")
    if spec.minimal_media_receipt is not None:
        return _freeze_iris_source_build_manifest_minimal(spec, output)
    source = _iris_source_checkout_receipt(spec.source_checkout)
    dependency_root = spec.dependency_root.resolve()
    dependency_tree = _path_tree_manifest(dependency_root)
    tools = _iris_trusted_build_tools(spec)
    archives: list[dict[str, object]] = []
    package_names: set[str] = set()
    for archive_ref in spec.dependency_archives:
        archive = archive_ref.resolve()
        if not archive.is_file() or archive.suffix != ".deb":
            raise ExternalLeagueError("EA IRIS dependency archive is missing or not a Debian package")
        completed = subprocess.run(
            [str(tools["dpkg_deb"]), "-f", str(archive), "Package", "Version", "Architecture"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        field_rows: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                field_rows[key] = value
        package = field_rows.get("Package", "")
        version = field_rows.get("Version", "")
        architecture = field_rows.get("Architecture", "")
        if completed.returncode != 0 or set(field_rows) != {"Package", "Version", "Architecture"} or architecture not in {"amd64", "all"}:
            raise ExternalLeagueError("EA IRIS dependency archive metadata is invalid")
        if package in package_names or not package or not version:
            raise ExternalLeagueError("EA IRIS dependency closure contains duplicate or empty package identity")
        package_names.add(package)
        archives.append({
            "package": package,
            "version": version,
            "architecture": architecture,
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        })
    observed_package_closure = {
        str(row["package"]): (
            str(row["version"]),
            str(row["architecture"]),
            str(row["sha256"]),
        )
        for row in archives
    }
    if observed_package_closure != EA_IRIS_SOURCE_DEBIAN_CLOSURE:
        raise ExternalLeagueError("EA IRIS Debian dependency closure differs from the frozen package version and SHA256 allowlist")
    sorted_archives = sorted(archives, key=lambda row: (str(row["package"]), str(row["version"])))
    with tempfile.TemporaryDirectory(prefix="flashpatch-iris-deb-closure-") as extracted_directory:
        extracted_root = Path(extracted_directory).resolve()
        for row in sorted_archives:
            completed = subprocess.run(
                [str(tools["dpkg_deb"]), "-x", str(row["path"]), str(extracted_root)],
                capture_output=True,
                check=False,
                timeout=120,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            )
            if completed.returncode != 0:
                raise ExternalLeagueError("EA IRIS dependency archive cannot be independently extracted")
        independently_extracted_tree = _path_tree_manifest(extracted_root)
    if independently_extracted_tree != dependency_tree:
        raise ExternalLeagueError(
            "EA IRIS dependency root is not the exact independent extraction of its frozen Debian archives"
        )
    required_dependency_paths = (
        "usr/include/opencv4/opencv2/core.hpp",
        "usr/include/opencv4/opencv2/imgproc.hpp",
        "usr/include/opencv4/opencv2/videoio.hpp",
        "usr/include/nlohmann/json.hpp",
        "usr/include/spdlog/spdlog.h",
        "usr/include/openssl/sha.h",
        "usr/include/x86_64-linux-gnu/openssl/opensslconf.h",
        "usr/lib/x86_64-linux-gnu/libopencv_core.so",
        "usr/lib/x86_64-linux-gnu/libopencv_imgproc.so",
        "usr/lib/x86_64-linux-gnu/libopencv_videoio.so",
        "usr/lib/x86_64-linux-gnu/libopencv_highgui.so",
        "usr/lib/x86_64-linux-gnu/libspdlog.so",
        "usr/lib/x86_64-linux-gnu/libfmt.so",
        "usr/lib/x86_64-linux-gnu/libavformat.so",
        "usr/lib/x86_64-linux-gnu/libavcodec.so",
        "usr/lib/x86_64-linux-gnu/libavutil.so",
        "usr/lib/x86_64-linux-gnu/libswscale.so",
    )
    if any(not (dependency_root / relative).exists() for relative in required_dependency_paths):
        raise ExternalLeagueError("EA IRIS extracted dependency closure is incomplete")
    tool_rows = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in tools.items()
    }
    compiler_version = subprocess.run(
        [str(tools["compiler"]), "--version"],
        capture_output=True,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    if compiler_version.returncode != 0:
        raise ExternalLeagueError("EA IRIS compiler version cannot be observed")
    build_environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "SOURCE_DATE_EPOCH": "1710356805",
        "ZERO_AR_DATE": "1",
    }
    adapter_hash = _sha256_bytes((_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n").encode("utf-8"))
    manifest = {
        "schema": "flashpatch-l7-ea-iris-source-build-manifest-v1",
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "distribution": "local-source-build-d96978ac-not-official-release",
        "source": source,
        "adapter": {
            "source_sha256": adapter_hash,
            "public_api_sequence": list(EA_IRIS_SOURCE_BOUNDARY_METHODS),
            "input_contract": "canonical-60fps-cfr-ffv1-rgb24-v1",
        },
        "dependency_closure": {
            "root": str(dependency_root),
            "archives": sorted_archives,
            "archives_sha256": _canonical_json_sha256(sorted_archives),
            "tree": dependency_tree,
            "tree_sha256": _canonical_json_sha256(dependency_tree),
            "extraction_relation": "EXACT_DPKG_DEB_X_REPLAY",
        },
        "toolchain": {
            **tool_rows,
            "compiler_version_stdout_sha256": _sha256_bytes(compiler_version.stdout),
            "compiler_version_stderr_sha256": _sha256_bytes(compiler_version.stderr),
        },
        "build_environment": build_environment,
        "build_environment_sha256": _canonical_json_sha256(build_environment),
        "scoreable": False,
        "scoreable_blockers": [
            "build_not_yet_executed",
            "release_oracle_conformance_not_yet_verified",
            "local_execution_witness_not_independent",
        ],
    }
    _write_json(output, manifest)
    return {**manifest, "manifest": str(output), "manifest_sha256": _sha256_file(output)}


def _load_iris_source_build_manifest_minimal(payload: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "schema", "identity", "distribution", "source", "adapter",
        "dependency_closure", "toolchain", "build_environment",
        "build_environment_sha256", "scoreable", "scoreable_blockers",
    }
    if set(payload) != expected_fields or payload.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID:
        raise ExternalLeagueError("EA IRIS minimal source build manifest identity or fields are invalid")
    if payload.get("distribution") != "local-source-build-d96978ac-not-official-release":
        raise ExternalLeagueError("EA IRIS minimal source build cannot claim official release identity")
    source = payload.get("source")
    adapter = payload.get("adapter")
    closure = payload.get("dependency_closure")
    toolchain = payload.get("toolchain")
    environment = payload.get("build_environment")
    if not all(isinstance(item, Mapping) for item in (source, adapter, closure, toolchain, environment)):
        raise ExternalLeagueError("EA IRIS minimal source build manifest sections are invalid")
    fresh_source = _iris_source_checkout_receipt(Path(str(source.get("path", ""))))
    if dict(source) != fresh_source:
        raise ExternalLeagueError("EA IRIS source tree drifted after minimal build manifest freeze")
    source_root = Path(str(source["path"])).resolve()
    pattern = source_root / "src" / "PatternDetection.cpp"
    requirement = EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS["src/PatternDetection.cpp"]
    video = source_root / "src" / "VideoAnalyser.cpp"
    video_requirement = EA_IRIS_MINIMAL_FFMPEG_SOURCE_REQUIREMENTS["src/VideoAnalyser.cpp"]
    video_text = video.read_text(encoding="utf-8")
    if (
        _sha256_file(pattern) != requirement["sha256"]
        or requirement["required_header"] not in pattern.read_text(encoding="utf-8")
        or _sha256_file(video) != video_requirement["sha256"]
        or any(header not in video_text for header in video_requirement["required_headers"])
        or any(call not in video_text for call in video_requirement["required_calls"])
        or "vA.AnalyseVideo" not in (source_root / "example" / "main.cpp").read_text(encoding="utf-8")
    ):
        raise ExternalLeagueError("EA IRIS source-required module evidence drifted after freeze")
    if (
        adapter.get("source_sha256") != _sha256_bytes((_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n").encode("utf-8"))
        or adapter.get("public_api_sequence") != list(EA_IRIS_SOURCE_BOUNDARY_METHODS)
        or adapter.get("input_contract") != "canonical-60fps-cfr-ffv1-rgb24-v1"
    ):
        raise ExternalLeagueError("EA IRIS generated adapter source or API boundary drifted")
    media_ref = closure.get("minimal_media_receipt")
    if (
        closure.get("kind") != "PINNED_MINIMAL_OPENCV_4_8_0_FFMPEG_N6_1_1_SOURCE_BUILD_V1"
        or closure.get("effective_opencv_modules") != list(EA_IRIS_MINIMAL_OPENCV_MODULES)
        or _canonical_json_sha256(closure.get("source_module_provenance"))
        != _canonical_json_sha256({
            "opencv": EA_IRIS_MINIMAL_OPENCV_SOURCE_REQUIREMENTS,
            "ffmpeg": EA_IRIS_MINIMAL_FFMPEG_SOURCE_REQUIREMENTS,
        })
        or not isinstance(media_ref, Mapping)
        or not isinstance(media_ref.get("path"), str)
    ):
        raise ExternalLeagueError("EA IRIS minimal media closure contract drifted")
    media_path = Path(str(media_ref["path"])).resolve()
    if not media_path.is_file() or media_ref.get("sha256") != _sha256_file(media_path):
        raise ExternalLeagueError("EA IRIS minimal media receipt hash drifted")
    media, _ = _load_iris_minimal_media_toolchain(media_path)
    dependency_root = Path(str(closure.get("root", ""))).resolve()
    if (
        dependency_root != Path(str(media["sdk"]["root"])).resolve()
        or closure.get("tree_sha256") != media["sdk"]["tree_sha256"]
        or closure.get("build_tool_closure") != media["build_tool_closure"]
    ):
        raise ExternalLeagueError("EA IRIS minimal media SDK binding drifted")
    if payload.get("build_environment_sha256") != _canonical_json_sha256(environment):
        raise ExternalLeagueError("EA IRIS minimal build environment hash mismatches")
    for name in ("compiler", "archiver", "dpkg_deb", "readelf", "ldd"):
        row = toolchain.get(name)
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ExternalLeagueError("EA IRIS minimal build toolchain ledger is invalid")
        tool = Path(str(row["path"])).resolve()
        expected_path, expected_sha256 = EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN[name]
        if (
            not tool.is_file()
            or tool != Path(expected_path).resolve()
            or row.get("path") != str(tool)
            or row.get("sha256") != expected_sha256
            or _sha256_file(tool) != expected_sha256
        ):
            raise ExternalLeagueError("EA IRIS minimal build toolchain binary drifted after freeze")
    if payload.get("scoreable") is not False:
        raise ExternalLeagueError("EA IRIS source build manifest cannot authorize scoring")
    return dict(payload)


def _load_iris_source_build_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source build manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ExternalLeagueError("EA IRIS source build manifest schema is invalid")
    if payload.get("schema") == "flashpatch-l7-ea-iris-source-build-manifest-v2":
        return _load_iris_source_build_manifest_minimal(payload)
    if payload.get("schema") != "flashpatch-l7-ea-iris-source-build-manifest-v1":
        raise ExternalLeagueError("EA IRIS source build manifest schema is invalid")
    expected_fields = {
        "schema", "identity", "distribution", "source", "adapter",
        "dependency_closure", "toolchain", "build_environment",
        "build_environment_sha256", "scoreable", "scoreable_blockers",
    }
    if set(payload) != expected_fields or payload.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID:
        raise ExternalLeagueError("EA IRIS source build manifest identity or fields are invalid")
    if payload.get("distribution") != "local-source-build-d96978ac-not-official-release":
        raise ExternalLeagueError("EA IRIS source build cannot claim official release identity")
    source = payload.get("source")
    adapter = payload.get("adapter")
    closure = payload.get("dependency_closure")
    toolchain = payload.get("toolchain")
    environment = payload.get("build_environment")
    if not all(isinstance(item, Mapping) for item in (source, adapter, closure, toolchain, environment)):
        raise ExternalLeagueError("EA IRIS source build manifest sections are invalid")
    fresh_source = _iris_source_checkout_receipt(Path(str(source.get("path", ""))))
    if dict(source) != fresh_source:
        raise ExternalLeagueError("EA IRIS source tree drifted after build manifest freeze")
    if (
        adapter.get("source_sha256") != _sha256_bytes((_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n").encode("utf-8"))
        or adapter.get("public_api_sequence") != list(EA_IRIS_SOURCE_BOUNDARY_METHODS)
        or adapter.get("input_contract") != "canonical-60fps-cfr-ffv1-rgb24-v1"
    ):
        raise ExternalLeagueError("EA IRIS generated adapter source or API boundary drifted")
    dependency_root = Path(str(closure.get("root", ""))).resolve()
    fresh_tree = _path_tree_manifest(dependency_root)
    if (
        set(closure) != {"root", "archives", "archives_sha256", "tree", "tree_sha256", "extraction_relation"}
        or closure.get("extraction_relation") != "EXACT_DPKG_DEB_X_REPLAY"
        or closure.get("tree") != fresh_tree
        or closure.get("tree_sha256") != _canonical_json_sha256(fresh_tree)
    ):
        raise ExternalLeagueError("EA IRIS extracted dependency closure drifted")
    archives = closure.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ExternalLeagueError("EA IRIS dependency archive ledger is missing")
    for row in archives:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ExternalLeagueError("EA IRIS dependency archive ledger is invalid")
        archive = Path(str(row["path"])).resolve()
        if not archive.is_file() or row.get("sha256") != _sha256_file(archive) or row.get("bytes") != archive.stat().st_size:
            raise ExternalLeagueError("EA IRIS dependency archive drifted after freeze")
    observed_package_closure = {
        str(row.get("package")): (
            str(row.get("version")),
            str(row.get("architecture")),
            str(row.get("sha256")),
        )
        for row in archives
    }
    if observed_package_closure != EA_IRIS_SOURCE_DEBIAN_CLOSURE:
        raise ExternalLeagueError("EA IRIS dependency archive ledger is outside the frozen package allowlist")
    if closure.get("archives_sha256") != _canonical_json_sha256(archives):
        raise ExternalLeagueError("EA IRIS dependency archive ledger hash mismatches")
    if payload.get("build_environment_sha256") != _canonical_json_sha256(environment):
        raise ExternalLeagueError("EA IRIS build environment hash mismatches")
    for name in ("compiler", "archiver", "dpkg_deb", "readelf", "ldd"):
        row = toolchain.get(name)
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ExternalLeagueError("EA IRIS toolchain ledger is invalid")
        tool = Path(str(row["path"])).resolve()
        expected_path, expected_sha256 = EA_IRIS_SOURCE_TRUSTED_TOOLCHAIN[name]
        if (
            not tool.is_file()
            or tool != Path(expected_path).resolve()
            or row.get("path") != str(tool)
            or row.get("sha256") != expected_sha256
            or _sha256_file(tool) != expected_sha256
        ):
            raise ExternalLeagueError("EA IRIS toolchain binary drifted after freeze")
    dpkg_deb = Path(str(toolchain["dpkg_deb"]["path"])).resolve()
    with tempfile.TemporaryDirectory(prefix="flashpatch-iris-manifest-replay-") as extracted_directory:
        extracted_root = Path(extracted_directory).resolve()
        for row in archives:
            completed = subprocess.run(
                [str(dpkg_deb), "-x", str(row["path"]), str(extracted_root)],
                capture_output=True,
                check=False,
                timeout=120,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            )
            if completed.returncode != 0:
                raise ExternalLeagueError("EA IRIS dependency manifest archive replay failed")
        replay_tree = _path_tree_manifest(extracted_root)
    if replay_tree != fresh_tree:
        raise ExternalLeagueError("EA IRIS dependency root differs from independent archive replay")
    if payload.get("scoreable") is not False:
        raise ExternalLeagueError("EA IRIS source build manifest cannot authorize scoring")
    return dict(payload)


def _run_iris_build_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_root: Path,
    ordinal: int,
) -> dict[str, object]:
    started = time.monotonic_ns()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        capture_output=True,
        check=False,
        timeout=600,
    )
    finished = time.monotonic_ns()
    stdout_path = log_root / f"command-{ordinal:03d}.stdout.bin"
    stderr_path = log_root / f"command-{ordinal:03d}.stderr.bin"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    row = {
        "ordinal": ordinal,
        "argv": list(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "wall_time_ns": finished - started,
        "stdout": {"path": stdout_path.relative_to(cwd).as_posix(), "sha256": _sha256_file(stdout_path)},
        "stderr": {"path": stderr_path.relative_to(cwd).as_posix(), "sha256": _sha256_file(stderr_path)},
    }
    if completed.returncode != 0:
        raise ExternalLeagueError(
            f"EA IRIS source build command {ordinal} failed; stderr_sha256={row['stderr']['sha256']}"
        )
    return row


def _iris_dynamic_closure(
    binary: Path,
    *,
    ldd: Path,
    library_root: Path,
    dependency_root: Path | None = None,
) -> dict[str, object]:
    completed = subprocess.run(
        [str(ldd), str(binary)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LD_LIBRARY_PATH": os.pathsep.join((str(library_root), str(library_root.parent))),
        },
    )
    if completed.returncode != 0 or "not found" in completed.stdout or "not found" in completed.stderr:
        raise ExternalLeagueError("EA IRIS built binary has an unresolved dynamic dependency")
    resolved: set[Path] = set()
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        match = re.search(r"=>\s+(/[^\s]+)\s+\(", line)
        if match is None:
            match = re.match(r"(/[^\s]+)\s+\(", line)
        if match is None:
            continue
        path = Path(match.group(1)).resolve()
        if not path.is_file():
            raise ExternalLeagueError("EA IRIS dynamic dependency path is not a file")
        resolved.add(path)
    if not resolved:
        raise ExternalLeagueError("EA IRIS dynamic dependency closure is empty")
    allowed_root = (
        dependency_root.resolve()
        if dependency_root is not None
        else library_root.parent.resolve()
    )
    system_abi_prefixes = (
        "ld-linux-x86-64.so.",
        "libc.so.",
        "libdl.so.",
        "libgcc_s.so.",
        "libm.so.",
        "libpthread.so.",
        "librt.so.",
        "libstdc++.so.",
    )
    host_fallbacks: list[str] = []
    for path in sorted(resolved):
        try:
            path.relative_to(allowed_root)
        except ValueError:
            if not path.name.startswith(system_abi_prefixes):
                host_fallbacks.append(str(path))
    if host_fallbacks:
        raise ExternalLeagueError(
            "EA IRIS dynamic closure used non-system host libraries outside the frozen Debian sysroot: "
            + ",".join(host_fallbacks)
        )
    rows: list[dict[str, object]] = []
    for path in sorted(resolved):
        try:
            relative = path.relative_to(allowed_root).as_posix()
        except ValueError:
            rows.append({
                "scope": "SYSTEM_ABI",
                "soname": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
        else:
            rows.append({
                "scope": "BUNDLE",
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    return {
        "ldd_observation_sha256": _canonical_json_sha256(rows),
        "libraries": rows,
        "libraries_sha256": _canonical_json_sha256(rows),
        "resolution_policy": "FROZEN_SYSROOT_OR_MINIMAL_SYSTEM_ABI_ONLY_V1",
        "system_abi_prefixes": list(system_abi_prefixes),
        "host_fallbacks": [],
    }


def _iris_source_link_flags(dependency_root: Path, *, direct_adapter: bool) -> list[str]:
    library_root = dependency_root / "lib"
    common = [
        f"-L{library_root}",
        "-Wl,--gc-sections",
        "-Wl,-z,relro",
        "-Wl,-z,now",
        "-Wl,--as-needed",
        "-Wl,--disable-new-dtags",
        "-Wl,-rpath,$ORIGIN/../lib",
        f"-Wl,-rpath-link,{library_root}",
    ]
    if direct_adapter:
        return [
            *common,
            "-lopencv_features2d",
            "-lopencv_imgproc",
            "-lopencv_core",
            "-lspdlog",
            "-lfmt",
            "-lcrypto",
            "-pthread",
            "-ldl",
        ]
    return [
        *common,
        "-lopencv_videoio",
        "-lopencv_imgcodecs",
        "-lopencv_features2d",
        "-lopencv_imgproc",
        "-lopencv_core",
        "-lspdlog",
        "-lfmt",
        "-lavformat",
        "-lavcodec",
        "-lavutil",
        "-lswscale",
        "-lcrypto",
        "-pthread",
        "-ldl",
    ]


def _iris_source_include_flags(
    source: Path,
    dependency_root: Path,
    *,
    minimal_media: bool,
) -> list[str]:
    if minimal_media:
        dependency_includes = (
            dependency_root / "include" / "opencv4",
            dependency_root / "include",
            dependency_root / "include" / "x86_64-linux-gnu",
        )
    else:
        dependency_includes = (
            dependency_root / "usr" / "include" / "opencv4",
            dependency_root / "usr" / "include",
            dependency_root / "usr" / "include" / "x86_64-linux-gnu",
        )
    return [
        f"-I{source / 'include'}",
        f"-I{source / 'src'}",
        f"-I{source / 'utils' / 'include'}",
        *[f"-I{path}" for path in dependency_includes],
    ]


def _audit_iris_direct_binary_boundary(binary: Path, *, readelf: Path) -> dict[str, object]:
    """Prove the direct child cannot silently retain a native video decoder."""
    dynamic = subprocess.run(
        [str(readelf), "-d", str(binary)],
        capture_output=True,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    symbols = subprocess.run(
        [str(readelf), "-Ws", str(binary)],
        capture_output=True,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    if dynamic.returncode != 0 or symbols.returncode != 0:
        raise ExternalLeagueError("EA IRIS direct adapter ELF boundary cannot be inspected")
    dynamic_text = dynamic.stdout.decode("utf-8", "replace")
    symbol_text = symbols.stdout.decode("utf-8", "replace")
    needed = sorted(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic_text))
    forbidden_needed = {
        "libopencv_videoio.so", "libopencv_imgcodecs.so", "libopencv_highgui.so",
        "libavcodec.so", "libavformat.so", "libavutil.so", "libswscale.so",
    }
    if any(any(name.startswith(prefix) for prefix in forbidden_needed) for name in needed):
        raise ExternalLeagueError("EA IRIS direct adapter retained a forbidden video decoder dependency")
    forbidden_symbol_fragments = (
        "VideoCapture", "avformat_", "avcodec_", "av_read_", "sws_",
    )
    if any(fragment in symbol_text for fragment in forbidden_symbol_fragments):
        raise ExternalLeagueError("EA IRIS direct adapter retained a forbidden video decoder symbol")
    if not any(name.startswith("libopencv_core.so") for name in needed):
        raise ExternalLeagueError("EA IRIS direct adapter does not bind the expected OpenCV analysis core")
    return {
        "contract": "RAW_RGB_ONLY_NO_NATIVE_VIDEO_DECODER_V1",
        "needed": needed,
        "dynamic_stdout_sha256": _sha256_bytes(dynamic.stdout),
        "dynamic_stderr_sha256": _sha256_bytes(dynamic.stderr),
        "symbols_stdout_sha256": _sha256_bytes(symbols.stdout),
        "symbols_stderr_sha256": _sha256_bytes(symbols.stderr),
        "forbidden_needed_absent": True,
        "forbidden_symbols_absent": True,
    }


def build_iris_source_frame_adapter(
    build_manifest: Path | str,
    output_root: Path | str,
) -> dict[str, object]:
    """Build the d969 adapter and its unmodified source-video oracle.

    This is intentionally a local source distribution, not EA's official
    1.1.0 release.  Every upstream translation unit is compiled without
    modification; only the separately generated adapter owns instrumentation.
    """
    manifest_path = Path(build_manifest).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS source build output already exists: {root}")
    manifest = _load_iris_source_build_manifest(manifest_path)
    source_section = manifest["source"]
    closure = manifest["dependency_closure"]
    toolchain = manifest["toolchain"]
    minimal_media = manifest.get("schema") == "flashpatch-l7-ea-iris-source-build-manifest-v2"
    source = Path(str(source_section["path"])).resolve()
    dependency_root = Path(str(closure["root"])).resolve()
    compiler = Path(str(toolchain["compiler"]["path"])).resolve()
    archiver = Path(str(toolchain["archiver"]["path"])).resolve()
    readelf = Path(str(toolchain["readelf"]["path"])).resolve()
    ldd = Path(str(toolchain["ldd"]["path"])).resolve()
    environment = dict(manifest["build_environment"])
    root.mkdir(parents=True)
    objects = root / "objects"
    logs = root / "logs"
    objects.mkdir()
    logs.mkdir()
    binary_root = root
    runtime_library_root = dependency_root / "usr" / "lib" / "x86_64-linux-gnu"
    if minimal_media:
        binary_root = root / "bin"
        runtime_library_root = root / "lib"
        binary_root.mkdir()
        runtime_library_root.mkdir()
        _copy_relative_symlink_tree(dependency_root / "lib", runtime_library_root)
    adapter_source = root / "ea-iris-source-frame-adapter.cpp"
    adapter_source.write_text(_EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n", encoding="utf-8")
    adapter_source_sha256 = _sha256_file(adapter_source)
    if adapter_source_sha256 != manifest["adapter"]["source_sha256"]:
        raise ExternalLeagueError("EA IRIS generated adapter source hash mismatches frozen manifest")
    configuration = root / "appsettings.json"
    shutil.copy2(source / "config" / "appsettings.json", configuration)
    license_artifact = root / "LICENSE.txt"
    shutil.copy2(source / "LICENSE.txt", license_artifact)
    if _sha256_file(configuration) != EA_IRIS_SOURCE_CONFIG_SHA256 or _sha256_file(license_artifact) != EA_IRIS_SOURCE_LICENSE_SHA256:
        raise ExternalLeagueError("EA IRIS staged configuration or license drifted")

    include_flags = _iris_source_include_flags(
        source,
        dependency_root,
        minimal_media=minimal_media,
    )
    common_flags = [
        "-std=c++17",
        "-O2",
        "-DNDEBUG",
        "-fstack-protector-strong",
        "-fno-omit-frame-pointer",
        "-ffunction-sections",
        "-fdata-sections",
        f"-ffile-prefix-map={source}=/usr/src/ea-iris-d96978ac",
        f"-ffile-prefix-map={root}=/build/flashpatch-ea-iris-source-adapter",
        *include_flags,
    ]
    command_rows: list[dict[str, object]] = []
    upstream_objects: list[Path] = []
    ordinal = 1
    for relative in EA_IRIS_SOURCE_CPP_PATHS:
        source_path = source / relative
        object_path = objects / (relative.replace("/", "__") + ".o")
        command = [
            str(compiler), *common_flags, f"-frandom-seed={relative}",
            "-c", str(source_path), "-o", str(object_path),
        ]
        command_rows.append(_run_iris_build_command(
            command, cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
        ))
        upstream_objects.append(object_path)
        ordinal += 1
    archive = root / "libiris-d96978ac.a"
    archive_command = [str(archiver), "rcsD", str(archive), *[str(path) for path in upstream_objects]]
    command_rows.append(_run_iris_build_command(
        archive_command, cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
    ))
    ordinal += 1
    adapter_object = objects / "flashpatch-adapter.o"
    adapter_define = f'-DFLASHPATCH_ADAPTER_SOURCE_SHA256="{adapter_source_sha256}"'
    command_rows.append(_run_iris_build_command(
        [
            str(compiler), *common_flags, adapter_define,
            "-frandom-seed=flashpatch-ea-iris-source-adapter",
            "-c", str(adapter_source), "-o", str(adapter_object),
        ],
        cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
    ))
    ordinal += 1
    source_video_object = objects / "upstream-example-main.o"
    command_rows.append(_run_iris_build_command(
        [
            str(compiler), *common_flags, "-frandom-seed=upstream-example-main",
            "-c", str(source / "example" / "main.cpp"), "-o", str(source_video_object),
        ],
        cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
    ))
    ordinal += 1
    adapter_link_flags = _iris_source_link_flags(dependency_root, direct_adapter=True)
    source_video_link_flags = _iris_source_link_flags(dependency_root, direct_adapter=False)
    adapter_binary = binary_root / "ea-iris-source-frame-adapter"
    command_rows.append(_run_iris_build_command(
        [str(compiler), str(adapter_object), str(archive), *adapter_link_flags, "-o", str(adapter_binary)],
        cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
    ))
    ordinal += 1
    source_video_binary = binary_root / "ea-iris-d969-source-video-oracle"
    command_rows.append(_run_iris_build_command(
        [str(compiler), str(source_video_object), str(archive), *source_video_link_flags, "-o", str(source_video_binary)],
        cwd=root, environment=environment, log_root=logs, ordinal=ordinal,
    ))
    expected_commands = _expected_iris_source_build_commands(manifest, root)
    if [row.get("argv") for row in command_rows] != expected_commands:
        raise ExternalLeagueError("EA IRIS executed build argv differs from the frozen trusted command plan")
    for binary in (adapter_binary, source_video_binary):
        binary.chmod(0o555)
    readelf_rows: dict[str, dict[str, object]] = {}
    dynamic_rows: dict[str, dict[str, object]] = {}
    for label, binary in (
        ("source_frame_adapter", adapter_binary),
        ("source_video_oracle", source_video_binary),
    ):
        completed = subprocess.run(
            [str(readelf), "-d", str(binary)],
            capture_output=True,
            check=False,
            timeout=30,
            env=environment,
        )
        expected_rpath = b"$ORIGIN/../lib" if minimal_media else str(runtime_library_root).encode("utf-8")
        if completed.returncode != 0 or expected_rpath not in completed.stdout:
            raise ExternalLeagueError("EA IRIS built binary does not bind the frozen dependency RPATH")
        readelf_rows[label] = {
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stderr_sha256": _sha256_bytes(completed.stderr),
        }
        dynamic_rows[label] = _iris_dynamic_closure(
            binary,
            ldd=ldd,
            library_root=runtime_library_root,
            dependency_root=root if minimal_media else dependency_root,
        )
    direct_boundary = _audit_iris_direct_binary_boundary(adapter_binary, readelf=readelf)
    fresh_dependency_tree = _path_tree_manifest(dependency_root)
    dependency_tree_matches = (
        _canonical_json_sha256(fresh_dependency_tree) == closure["tree_sha256"]
        if minimal_media
        else fresh_dependency_tree == closure["tree"]
    )
    if not dependency_tree_matches:
        raise ExternalLeagueError("EA IRIS build mutated its frozen dependency closure")
    fresh_source = _iris_source_checkout_receipt(source)
    if fresh_source != source_section:
        raise ExternalLeagueError("EA IRIS build mutated the frozen upstream source tree")
    binaries = {
        "source_frame_adapter": {
            "path": adapter_binary.relative_to(root).as_posix(),
            "bytes": adapter_binary.stat().st_size,
            "sha256": _sha256_file(adapter_binary),
        },
        "source_video_oracle": {
            "path": source_video_binary.relative_to(root).as_posix(),
            "bytes": source_video_binary.stat().st_size,
            "sha256": _sha256_file(source_video_binary),
        },
    }
    staged_manifest = root / "build-manifest.json"
    shutil.copy2(manifest_path, staged_manifest)
    if _sha256_file(staged_manifest) != _sha256_file(manifest_path):
        raise ExternalLeagueError("EA IRIS staged build manifest drifted")
    if minimal_media:
        media_ref = closure["minimal_media_receipt"]
        media, _ = _load_iris_minimal_media_toolchain(Path(str(media_ref["path"])))
        dependency_receipt = {
            "kind": closure["kind"],
            "minimal_media_receipt_sha256": media_ref["sha256"],
            "tree_sha256": closure["tree_sha256"],
            "runtime_bundle_tree": _path_tree_manifest(runtime_library_root),
            "dynamic": dynamic_rows,
        }
        sbom = {
            "format": "receipt-bound-source-archive-support-package-and-dynamic-library-ledger-v2",
            "sources": media["sources"],
            "packages": media["support_archives"],
            "build_tool_archives": media["build_tool_archives"],
            "build_tool_closure": media["build_tool_closure"],
            "runtime_libraries": dynamic_rows,
        }
    else:
        dependency_receipt = {
            "archives_sha256": closure["archives_sha256"],
            "tree_sha256": closure["tree_sha256"],
            "dynamic": dynamic_rows,
        }
        sbom = {
            "format": "receipt-bound-debian-archive-and-dynamic-library-ledger-v1",
            "packages": closure["archives"],
            "runtime_libraries": dynamic_rows,
        }
    receipt = {
        "schema": EA_IRIS_SOURCE_BUILD_SCHEMA_V2 if minimal_media else EA_IRIS_SOURCE_BUILD_SCHEMA,
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "distribution": "local-source-build-d96978ac-not-official-release",
        "source": source_section,
        "build_manifest": {"path": staged_manifest.name, "sha256": _sha256_file(staged_manifest)},
        "adapter_source": {"path": adapter_source.name, "sha256": adapter_source_sha256},
        "configuration": {"path": configuration.name, "sha256": _sha256_file(configuration)},
        "license": {"path": license_artifact.name, "sha256": _sha256_file(license_artifact), "spdx": "BSD-3-Clause"},
        "dependency_closure": dependency_receipt,
        "sbom": sbom,
        "toolchain": toolchain,
        "build_environment": environment,
        "build_environment_sha256": _canonical_json_sha256(environment),
        "commands": command_rows,
        "commands_sha256": _canonical_json_sha256(command_rows),
        "readelf": readelf_rows,
        "direct_adapter_boundary": direct_boundary,
        "binaries": binaries,
        "status": "BUILD_VERIFIED",
        "release_equivalence": "NOT_CLAIMED",
        "scoreable": False,
        "scoreable_blockers": [
            "source_video_conformance_not_yet_verified",
            "temporal_boundary_conformance_not_yet_verified",
            "local_execution_witness_not_independent",
            "independent_gold_receipt_missing",
            *(["minimal_media_host_coreutils_archive_not_closed"] if minimal_media else []),
        ],
    }
    receipt_path = root / "ea-iris-source-build-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def _expected_iris_source_build_commands(
    manifest: Mapping[str, object],
    root: Path,
) -> list[list[str]]:
    source = Path(str(manifest["source"]["path"])).resolve()
    dependency_root = Path(str(manifest["dependency_closure"]["root"])).resolve()
    compiler = Path(str(manifest["toolchain"]["compiler"]["path"])).resolve()
    archiver = Path(str(manifest["toolchain"]["archiver"]["path"])).resolve()
    objects = root / "objects"
    adapter_source = root / "ea-iris-source-frame-adapter.cpp"
    adapter_source_sha256 = str(manifest["adapter"]["source_sha256"])
    minimal_media = manifest.get("schema") == "flashpatch-l7-ea-iris-source-build-manifest-v2"
    include_flags = _iris_source_include_flags(
        source,
        dependency_root,
        minimal_media=minimal_media,
    )
    common_flags = [
        "-std=c++17", "-O2", "-DNDEBUG", "-fstack-protector-strong",
        "-fno-omit-frame-pointer", "-ffunction-sections", "-fdata-sections",
        f"-ffile-prefix-map={source}=/usr/src/ea-iris-d96978ac",
        f"-ffile-prefix-map={root}=/build/flashpatch-ea-iris-source-adapter",
        *include_flags,
    ]
    upstream_objects = [objects / (relative.replace("/", "__") + ".o") for relative in EA_IRIS_SOURCE_CPP_PATHS]
    commands = [
        [
            str(compiler), *common_flags, f"-frandom-seed={relative}",
            "-c", str(source / relative), "-o", str(object_path),
        ]
        for relative, object_path in zip(EA_IRIS_SOURCE_CPP_PATHS, upstream_objects)
    ]
    archive = root / "libiris-d96978ac.a"
    commands.append([str(archiver), "rcsD", str(archive), *[str(path) for path in upstream_objects]])
    adapter_object = objects / "flashpatch-adapter.o"
    commands.append([
        str(compiler), *common_flags,
        f'-DFLASHPATCH_ADAPTER_SOURCE_SHA256="{adapter_source_sha256}"',
        "-frandom-seed=flashpatch-ea-iris-source-adapter",
        "-c", str(adapter_source), "-o", str(adapter_object),
    ])
    source_video_object = objects / "upstream-example-main.o"
    commands.append([
        str(compiler), *common_flags, "-frandom-seed=upstream-example-main",
        "-c", str(source / "example" / "main.cpp"), "-o", str(source_video_object),
    ])
    binary_root = root / "bin" if minimal_media else root
    commands.append([
        str(compiler), str(adapter_object), str(archive),
        *_iris_source_link_flags(dependency_root, direct_adapter=True),
        "-o", str(binary_root / "ea-iris-source-frame-adapter"),
    ])
    commands.append([
        str(compiler), str(source_video_object), str(archive),
        *_iris_source_link_flags(dependency_root, direct_adapter=False),
        "-o", str(binary_root / "ea-iris-d969-source-video-oracle"),
    ])
    return commands


def _load_iris_source_build_receipt(receipt_ref: Path | str) -> tuple[dict[str, object], Path]:
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source build receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") not in {EA_IRIS_SOURCE_BUILD_SCHEMA, EA_IRIS_SOURCE_BUILD_SCHEMA_V2}
        or payload.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID
        or payload.get("distribution") != "local-source-build-d96978ac-not-official-release"
        or payload.get("status") != "BUILD_VERIFIED"
        or payload.get("release_equivalence") != "NOT_CLAIMED"
        or payload.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS source build receipt identity or terminal state is invalid")
    root = receipt_path.parent
    minimal_media = payload.get("schema") == EA_IRIS_SOURCE_BUILD_SCHEMA_V2
    manifest_ref = payload.get("build_manifest")
    if not isinstance(manifest_ref, Mapping) or not isinstance(manifest_ref.get("path"), str):
        raise ExternalLeagueError("EA IRIS source build receipt omits its frozen manifest")
    manifest_path = (root / str(manifest_ref["path"])).resolve()
    if not manifest_path.is_file() or manifest_ref.get("sha256") != _sha256_file(manifest_path):
        raise ExternalLeagueError("EA IRIS source build manifest hash drifted")
    manifest = _load_iris_source_build_manifest(manifest_path)
    if payload.get("source") != manifest["source"]:
        raise ExternalLeagueError("EA IRIS source build receipt source differs from manifest")
    for field, expected_hash in (
        ("adapter_source", manifest["adapter"]["source_sha256"]),
        ("configuration", EA_IRIS_SOURCE_CONFIG_SHA256),
        ("license", EA_IRIS_SOURCE_LICENSE_SHA256),
    ):
        reference = payload.get(field)
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ExternalLeagueError(f"EA IRIS source build receipt omits {field}")
        artifact = (root / str(reference["path"])).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError(f"EA IRIS source build {field} escapes build root") from exc
        if not artifact.is_file() or reference.get("sha256") != expected_hash or _sha256_file(artifact) != expected_hash:
            raise ExternalLeagueError(f"EA IRIS source build {field} hash drifted")
    adapter_source = root / str(payload["adapter_source"]["path"])
    if adapter_source.read_text(encoding="utf-8") != _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP + "\n":
        raise ExternalLeagueError("EA IRIS source adapter bytes differ from frozen generated source")
    binaries = payload.get("binaries")
    if not isinstance(binaries, Mapping) or set(binaries) != {"source_frame_adapter", "source_video_oracle"}:
        raise ExternalLeagueError("EA IRIS source build binary set is invalid")
    resolved_binaries: dict[str, Path] = {}
    for name, reference in binaries.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ExternalLeagueError("EA IRIS source build binary receipt is invalid")
        binary = (root / str(reference["path"])).resolve()
        try:
            binary.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError("EA IRIS source build binary escapes build root") from exc
        if (
            not binary.is_file()
            or not os.access(binary, os.X_OK)
            or reference.get("bytes") != binary.stat().st_size
            or reference.get("sha256") != _sha256_file(binary)
        ):
            raise ExternalLeagueError("EA IRIS source build binary hash or mode drifted")
        resolved_binaries[str(name)] = binary
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != len(EA_IRIS_SOURCE_CPP_PATHS) + 5:
        raise ExternalLeagueError("EA IRIS source build command ledger is incomplete")
    if payload.get("commands_sha256") != _canonical_json_sha256(commands):
        raise ExternalLeagueError("EA IRIS source build command ledger hash mismatches")
    expected_commands = _expected_iris_source_build_commands(manifest, root)
    if [row.get("argv") if isinstance(row, Mapping) else None for row in commands] != expected_commands:
        raise ExternalLeagueError("EA IRIS source build command ledger differs from the frozen trusted argv")
    for expected_ordinal, row in enumerate(commands, start=1):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != expected_ordinal
            or row.get("exit_code") != 0
            or row.get("cwd") != str(root)
            or isinstance(row.get("wall_time_ns"), bool)
            or not isinstance(row.get("wall_time_ns"), int)
            or row.get("wall_time_ns") <= 0
        ):
            raise ExternalLeagueError("EA IRIS source build command ledger is malformed")
        argv = row.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
            raise ExternalLeagueError("EA IRIS source build argv ledger is invalid")
        for stream in ("stdout", "stderr"):
            reference = row.get(stream)
            if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
                raise ExternalLeagueError("EA IRIS source build command log is missing")
            artifact = (root / str(reference["path"])).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as exc:
                raise ExternalLeagueError("EA IRIS source build command log escapes root") from exc
            if not artifact.is_file() or reference.get("sha256") != _sha256_file(artifact):
                raise ExternalLeagueError("EA IRIS source build command log hash drifted")
    closure = payload.get("dependency_closure")
    if not isinstance(closure, Mapping):
        raise ExternalLeagueError("EA IRIS source build dependency closure receipt is invalid")
    manifest_closure = manifest["dependency_closure"]
    if minimal_media:
        media_ref = manifest_closure.get("minimal_media_receipt")
        runtime_library_root = root / "lib"
        if (
            closure.get("kind") != manifest_closure.get("kind")
            or not isinstance(media_ref, Mapping)
            or closure.get("minimal_media_receipt_sha256") != media_ref.get("sha256")
            or closure.get("tree_sha256") != manifest_closure.get("tree_sha256")
            or closure.get("runtime_bundle_tree") != _path_tree_manifest(runtime_library_root)
        ):
            raise ExternalLeagueError("EA IRIS minimal dependency closure differs from manifest or bundle")
    else:
        if (
            closure.get("archives_sha256") != manifest_closure["archives_sha256"]
            or closure.get("tree_sha256") != manifest_closure["tree_sha256"]
        ):
            raise ExternalLeagueError("EA IRIS source build dependency closure differs from manifest")
    dynamic = closure.get("dynamic")
    if not isinstance(dynamic, Mapping) or set(dynamic) != set(resolved_binaries):
        raise ExternalLeagueError("EA IRIS source build dynamic closure set is invalid")
    manifest_dependency_root = Path(str(manifest_closure["root"])).resolve()
    library_root = root / "lib" if minimal_media else manifest_dependency_root / "usr" / "lib" / "x86_64-linux-gnu"
    ldd_row = manifest["toolchain"].get("ldd")
    if not isinstance(ldd_row, Mapping) or not isinstance(ldd_row.get("path"), str):
        raise ExternalLeagueError("EA IRIS source build ldd authority is missing")
    ldd_binary = Path(str(ldd_row["path"])).resolve()
    for label, dynamic_row in dynamic.items():
        if not isinstance(dynamic_row, Mapping) or not isinstance(dynamic_row.get("libraries"), list):
            raise ExternalLeagueError("EA IRIS source build dynamic library ledger is invalid")
        libraries = dynamic_row["libraries"]
        if dynamic_row.get("libraries_sha256") != _canonical_json_sha256(libraries):
            raise ExternalLeagueError("EA IRIS source build dynamic library ledger hash mismatches")
        for library in libraries:
            if not isinstance(library, Mapping):
                raise ExternalLeagueError("EA IRIS dynamic library identity is invalid")
            if library.get("scope") == "BUNDLE" and isinstance(library.get("path"), str):
                path = (root / str(library["path"])).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise ExternalLeagueError("EA IRIS dynamic bundle library escapes root") from exc
            elif library.get("scope") == "SYSTEM_ABI" and isinstance(library.get("soname"), str):
                continue
            elif not minimal_media and isinstance(library.get("path"), str):
                path = Path(str(library["path"])).resolve()
            else:
                raise ExternalLeagueError("EA IRIS dynamic library scope is invalid")
            if (
                not path.is_file()
                or library.get("bytes") != path.stat().st_size
                or library.get("sha256") != _sha256_file(path)
            ):
                raise ExternalLeagueError("EA IRIS dynamic dependency drifted after build")
        fresh_dynamic = _iris_dynamic_closure(
            resolved_binaries[str(label)],
            ldd=ldd_binary,
            library_root=library_root,
            dependency_root=root if minimal_media else manifest_dependency_root,
        )
        if dict(dynamic_row) != fresh_dynamic:
            raise ExternalLeagueError("EA IRIS dynamic dependency receipt differs from fresh strict replay")
    boundary = payload.get("direct_adapter_boundary")
    readelf_row = manifest["toolchain"].get("readelf")
    if not isinstance(boundary, Mapping) or not isinstance(readelf_row, Mapping):
        raise ExternalLeagueError("EA IRIS direct adapter ELF boundary receipt is missing")
    fresh_boundary = _audit_iris_direct_binary_boundary(
        resolved_binaries["source_frame_adapter"],
        readelf=Path(str(readelf_row.get("path", ""))).resolve(),
    )
    if dict(boundary) != fresh_boundary:
        raise ExternalLeagueError("EA IRIS direct adapter ELF boundary differs from fresh inspection")
    if payload.get("toolchain") != manifest["toolchain"] or payload.get("build_environment") != manifest["build_environment"]:
        raise ExternalLeagueError("EA IRIS source build toolchain or environment differs from manifest")
    return dict(payload), receipt_path


def _iris_built_binary(
    build_receipt_path: Path,
    build: Mapping[str, object],
    name: str,
) -> Path:
    binaries = build.get("binaries")
    if not isinstance(binaries, Mapping):
        raise ExternalLeagueError("EA IRIS source build binary ledger is missing")
    reference = binaries.get(name)
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        raise ExternalLeagueError("EA IRIS source build binary reference is invalid")
    binary = (build_receipt_path.parent / str(reference["path"])).resolve()
    try:
        binary.relative_to(build_receipt_path.parent)
    except ValueError as exc:
        raise ExternalLeagueError("EA IRIS source build binary escapes receipt root") from exc
    if not binary.is_file() or reference.get("sha256") != _sha256_file(binary):
        raise ExternalLeagueError("EA IRIS source build binary bytes drifted")
    return binary


def _parse_iris_source_adapter_output(
    output_path: Path,
    *,
    video: Path,
    conversion: Path,
    build: Mapping[str, object],
    direct_rgb: Mapping[str, object],
    execution_configuration: Path,
    expected_thread_limit: int,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source adapter raw ledger is unreadable") from exc
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if contract["fps"] != 60:
        raise ExternalLeagueError("EA IRIS source adapter accepts only canonical 60fps CFR")
    expected_fields = {
        "schema", "identity", "source_revision", "adapter_source_sha256", "input",
        "configuration", "decoder", "api_sequence", "frame_count", "frames",
        "prediction", "warning", "runtime_timing_eligible", "scoreable", "scoreable_blockers",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ExternalLeagueError("EA IRIS source adapter raw ledger fields are invalid")
    adapter_source = build.get("adapter_source")
    configuration = build.get("configuration")
    if (
        payload.get("schema") != "flashpatch-l7-ea-iris-source-child-adapter-v1"
        or payload.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID
        or payload.get("source_revision") != EA_IRIS_SOURCE_REVISION
        or not isinstance(adapter_source, Mapping)
        or payload.get("adapter_source_sha256") != adapter_source.get("sha256")
        or not isinstance(configuration, Mapping)
        or payload.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS source adapter identity or build binding drifted")
    input_row = payload.get("input")
    config_row = payload.get("configuration")
    decoder = payload.get("decoder")
    expected_decoder_fields = {
        "api", "backend", "reported_fps", "reported_frame_count", "required_cfr",
        "decoded_pixel_format", "analysis_thread_limit", "analysis_threads_observed",
        "decoder_thread_control", "runtime_boundary", "binary", "binary_sha256", "command",
    }
    decoder_thread_control = decoder.get("decoder_thread_control") if isinstance(decoder, Mapping) else None
    timeline = direct_rgb.get("payload")
    raw_ref = direct_rgb.get("raw_rgb")
    timeline_ref = direct_rgb.get("timeline")
    decoder_thread_fields_valid = (
        isinstance(decoder, Mapping)
        and set(decoder) == expected_decoder_fields
        and isinstance(timeline, Mapping)
        and decoder.get("api") == "pinned_ffmpeg_raw_rgb24_prematerialization"
        and decoder.get("backend") == "FFMPEG_COMMAND_BOUND_IN_TIMELINE"
        and decoder.get("binary") == timeline["decoder"]["binary"]
        and decoder.get("binary_sha256") == timeline["decoder"]["binary_sha256"]
        and decoder.get("command") == timeline["decoder"]["command"]
        and decoder.get("reported_fps") == 60
        and decoder.get("reported_frame_count") == contract["frame_count"]
        and decoder.get("required_cfr") == {"numerator": 60, "denominator": 1}
        and decoder.get("decoded_pixel_format") == "rgb24"
        and decoder.get("analysis_thread_limit") == expected_thread_limit
        and decoder.get("analysis_threads_observed") == expected_thread_limit
        and decoder_thread_control == "UNSUPPORTED_NOT_VERIFIED"
        and decoder.get("runtime_boundary") == "FFMPEG_PREMATERIALIZATION_EXCLUDED_FROM_MEASURED_CHILD"
    )
    expected_input = (
        {
            "raw_rgb": dict(raw_ref),
            "timeline": dict(timeline_ref),
            "source_video": timeline["source_video"],
            "conversion_receipt": timeline["conversion_receipt"],
            "renderer_source": timeline["renderer_source"],
        }
        if all(isinstance(item, Mapping) for item in (timeline, raw_ref, timeline_ref))
        else None
    )
    if (
        input_row != expected_input
        or not isinstance(config_row, Mapping)
        or Path(str(config_row.get("path", ""))).resolve() != execution_configuration
        or config_row.get("sha256") != configuration.get("sha256")
        or config_row.get("overrides") != {"pattern_detection": True, "frame_resize": False}
        or not decoder_thread_fields_valid
        or payload.get("api_sequence") != list(EA_IRIS_SOURCE_BOUNDARY_METHODS)
        or payload.get("frame_count") != contract["frame_count"]
        or payload.get("runtime_timing_eligible") is not False
    ):
        raise ExternalLeagueError("EA IRIS source adapter input, configuration, decoder, or API sequence drifted")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != contract["frame_count"]:
        raise ExternalLeagueError("EA IRIS source adapter frame ledger is incomplete")
    decoded_rgb, _, _ = _decode_canonical_video_rgb(video, conversion)
    hazardous_indices: list[int] = []
    warning_indices: list[int] = []
    expected_frame_fields = {
        "frame_index", "iris_frame_number", "cfr_timestamp", "cfr_timestamp_us_rounded",
        "renderer_timestamp_us", "iris_timestamp_ms", "shape", "rgb_pixel_format", "bgr_pixel_format",
        "rgb_sha256", "pre_analyse_rgb_sha256", "pre_analyse_bgr_sha256", "post_analyse_bgr_sha256",
        "native_frame_data",
    }
    expected_native_fields = {
        "luminance_average", "luminance_flash_area", "average_luminance_diff",
        "average_luminance_diff_acc", "red_average", "red_flash_area",
        "average_red_diff", "average_red_diff_acc", "pattern_risk",
        "luminance_transitions", "red_transitions", "luminance_extended_fail_count",
        "red_extended_fail_count", "luminance_result", "red_result", "pattern_area",
        "pattern_detected_lines", "pattern_result",
    }
    flash_names = {0: "Pass", 1: "PassWithWarning", 2: "ExtendedFail", 3: "FlashFail"}
    pattern_names = {0: "Pass", 1: "Fail"}
    for index, row in enumerate(frames):
        if not isinstance(row, Mapping) or set(row) != expected_frame_fields:
            raise ExternalLeagueError("EA IRIS source adapter frame ledger fields are invalid")
        canonical = contract["frame_map"][index]
        bgr = cv2.cvtColor(decoded_rgb[index], cv2.COLOR_RGB2BGR)
        expected_bgr_hash = _sha256_bytes(np.ascontiguousarray(bgr).tobytes())
        expected_timestamp_us = (index * 1_000_000 + 30) // 60
        if (
            row.get("frame_index") != index
            or row.get("iris_frame_number") != index + 1
            or row.get("cfr_timestamp") != {"numerator": index, "denominator": 60}
            or row.get("cfr_timestamp_us_rounded") != expected_timestamp_us
            or row.get("cfr_timestamp_us_rounded") != canonical["cfr_timestamp_us"]
            or row.get("renderer_timestamp_us") != canonical["renderer_timestamp_us"]
            or row.get("iris_timestamp_ms") != (index * 1000) // 60
            or row.get("shape") != list(decoded_rgb.shape[1:])
            or row.get("rgb_pixel_format") != "rgb24"
            or row.get("bgr_pixel_format") != "CV_8UC3"
            or row.get("rgb_sha256") != canonical["rgb_sha256"]
            or row.get("pre_analyse_rgb_sha256") != canonical["rgb_sha256"]
            or row.get("pre_analyse_bgr_sha256") != expected_bgr_hash
            or row.get("post_analyse_bgr_sha256") != expected_bgr_hash
        ):
            raise ExternalLeagueError("EA IRIS source adapter RGB/BGR or exact timeline ledger drifted")
        native = row.get("native_frame_data")
        if not isinstance(native, Mapping) or set(native) != expected_native_fields:
            raise ExternalLeagueError("EA IRIS source adapter native FrameData ledger is incomplete")
        luminance = native.get("luminance_result")
        red = native.get("red_result")
        pattern = native.get("pattern_result")
        if (
            not isinstance(luminance, Mapping)
            or not isinstance(red, Mapping)
            or not isinstance(pattern, Mapping)
            or luminance.get("name") != flash_names.get(luminance.get("code"))
            or red.get("name") != flash_names.get(red.get("code"))
            or pattern.get("name") != pattern_names.get(pattern.get("code"))
        ):
            raise ExternalLeagueError("EA IRIS source adapter native categorical result is invalid")
        numeric_fields = (
            "luminance_average", "average_luminance_diff", "average_luminance_diff_acc",
            "red_average", "average_red_diff", "average_red_diff_acc", "pattern_risk",
            "luminance_transitions", "red_transitions", "luminance_extended_fail_count",
            "red_extended_fail_count", "pattern_detected_lines",
        )
        if any(
            isinstance(native.get(field), bool)
            or not isinstance(native.get(field), (int, float))
            or not np.isfinite(native.get(field))
            for field in numeric_fields
        ):
            raise ExternalLeagueError("EA IRIS source adapter native numeric FrameData is invalid")
        if any(
            not isinstance(native.get(field), str)
            for field in ("luminance_flash_area", "red_flash_area", "pattern_area")
        ):
            raise ExternalLeagueError("EA IRIS source adapter native area result is invalid")
        if int(luminance["code"]) >= 2 or int(red["code"]) >= 2 or int(pattern["code"]) >= 1:
            hazardous_indices.append(index)
        if int(luminance["code"]) == 1 or int(red["code"]) == 1:
            warning_indices.append(index)
    prediction = "HAZARDOUS" if hazardous_indices else "SAFE"
    if payload.get("prediction") != prediction or payload.get("warning") != bool(warning_indices):
        raise ExternalLeagueError("EA IRIS source adapter terminal prediction conflicts with FrameData")
    normalized = {
        "tool": EA_IRIS_SOURCE_ADAPTER_ID,
        "distribution": "local-source-build-d96978ac-not-official-release",
        "prediction": prediction,
        "frame_count": contract["frame_count"],
        "fps": 60,
        "hazard_frame_indices": hazardous_indices,
        "warning_frame_indices": warning_indices,
        "warning": bool(warning_indices),
        "timestamp_metrics": "exact-cfr-rational-plus-rounded-microseconds",
        "mask_metrics": "NOT_APPLICABLE",
        "decoder_thread_control": decoder_thread_control,
        "runtime_timing_eligible": False,
        "runtime_timing_reason": "decoder_prematerialization_excluded_and_thread_control_not_verified",
    }
    return dict(payload), normalized


def execute_iris_source_frame_adapter(
    build_receipt: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    census_receipt: Path | str,
    census_artifact_root: Path | str,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    scheduled_repeat_ordinal: int,
    runtime_schedule: Path | str,
    schedule_slot: int,
    execution_lane: str = "DIRECT_DETECTOR",
    conformance_receipt: Path | str | None = None,
) -> dict[str, object]:
    """Run the frozen d969 adapter as one scheduled local-witness child."""
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS source adapter output already exists: {root}")
    build, build_path = _load_iris_source_build_receipt(build_receipt)
    if execution_lane not in {"DIRECT_DETECTOR", "CONFORMANCE_ONLY"}:
        raise ExternalLeagueError("EA IRIS source adapter execution lane is invalid")
    conformance_binding: dict[str, object] | None = None
    if execution_lane == "DIRECT_DETECTOR":
        if conformance_receipt is None:
            raise ExternalLeagueError("EA IRIS direct detector execution requires source/release conformance")
        conformance, conformance_path = _load_iris_source_conformance_receipt(conformance_receipt)
        if conformance.get("source_build") != {"path": str(build_path), "sha256": _sha256_file(build_path), "binary_sha256": build["binaries"]["source_frame_adapter"]["sha256"]}:
            raise ExternalLeagueError("EA IRIS direct detector build differs from conformance authority")
        conformance_binding = {"path": str(conformance_path), "sha256": _sha256_file(conformance_path)}
    elif conformance_receipt is not None:
        raise ExternalLeagueError("EA IRIS conformance-only execution cannot claim prior conformance")
    census_entry, census_path = _load_execution_census_entry(
        census_receipt,
        census_artifact_root,
        EA_IRIS_SOURCE_ADAPTER_ID,
    )
    if (
        census_entry.get("repository_url") != build["source"]["repository_url"]
        or census_entry.get("revision") != build["source"]["revision"]
        or census_entry.get("license") != "BSD-3-Clause"
    ):
        raise ExternalLeagueError("EA IRIS source adapter build differs from frozen census identity")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if contract["fps"] != 60:
        raise ExternalLeagueError("EA IRIS source adapter rejects non-60fps canonical input")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen_runtime is None:
        raise ExternalLeagueError("EA IRIS source adapter requires a frozen fair runtime protocol")
    schedule_binding = _load_schedule_assignment(
        runtime_schedule,
        schedule_slot=schedule_slot,
        protocol=frozen_runtime,
        comparator=EA_IRIS_SOURCE_ADAPTER_ID,
        repeat_ordinal=scheduled_repeat_ordinal,
        input_sha256=_sha256_file(video),
    )
    adapter_binary = _iris_built_binary(build_path, build, "source_frame_adapter")
    build_configuration = (build_path.parent / str(build["configuration"]["path"])).resolve()
    root.mkdir(parents=True)
    direct_rgb = _materialize_iris_direct_rgb_input(video, conversion, root)
    raw_rgb = Path(str(direct_rgb["raw_rgb"]["path"])).resolve()
    timeline = Path(str(direct_rgb["timeline"]["path"])).resolve()
    configuration_root = root / "configuration"
    configuration_root.mkdir()
    configuration = configuration_root / "appsettings.json"
    shutil.copy2(build_configuration, configuration)
    configuration.chmod(0o444)
    if _sha256_file(configuration) != build["configuration"]["sha256"]:
        raise ExternalLeagueError("EA IRIS private staged configuration differs from source build")
    raw_output = root / "adapter-output.json"
    config_directory = str(configuration.parent) + os.sep
    _, height, width, channels = contract["shape"]
    if channels != 3:
        raise ExternalLeagueError("EA IRIS direct RGB adapter requires three-channel RGB")
    tool_command = [
        str(adapter_binary),
        str(raw_rgb),
        str(timeline),
        config_directory,
        str(raw_output),
        str(width),
        str(height),
        str(contract["frame_count"]),
        str(direct_rgb["raw_rgb"]["sha256"]),
        str(direct_rgb["timeline"]["sha256"]),
        _sha256_file(configuration),
        str(frozen_runtime["threads"]["limit"]),
    ]
    with _fair_execution_context(
        frozen_runtime,
        raw_rgb,
        base_environment=os.environ,
        schedule_binding=schedule_binding,
    ) as execution:
        probe_path = root / "runtime-probe.json"
        command = _instrument_fair_command(execution, tool_command, probe_path)
        started = time.monotonic_ns()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=execution["environment"],
                capture_output=True,
                check=False,
                timeout=int(frozen_runtime["budget"]["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: EA IRIS source adapter timeout",
            )
        stdout_path = root / "stdout.bin"
        stderr_path = root / "stderr.bin"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        raw_payload: dict[str, object] | None = None
        observation: dict[str, object] | None = None
        parse_error: str | None = None
        if completed.returncode == 0 and raw_output.is_file():
            try:
                raw_payload, observation = _parse_iris_source_adapter_output(
                    raw_output,
                    video=video,
                    conversion=conversion,
                    build=build,
                    direct_rgb=direct_rgb,
                    execution_configuration=configuration,
                    expected_thread_limit=int(frozen_runtime["threads"]["limit"]),
                )
            except ExternalLeagueError as exc:
                parse_error = str(exc)
        child_probe: dict[str, object] | None = None
        try:
            child_probe = _load_child_runtime_probe(probe_path)
        except ExternalLeagueError:
            pass
        finished = time.monotonic_ns()
        elapsed_ns = finished - started
        if elapsed_ns > int(frozen_runtime["budget"]["timeout_seconds"]) * 1_000_000_000:
            timed_out = True
        process_valid = (
            completed.returncode == 0
            and not timed_out
            and raw_payload is not None
            and observation is not None
            and child_probe is not None
        )
        runtime_receipt = _fair_runtime_run_receipt(
            frozen_runtime,
            comparator=EA_IRIS_SOURCE_ADAPTER_ID,
            scheduled_repeat_ordinal=scheduled_repeat_ordinal,
            schedule_binding=schedule_binding,
            input_sha256=_sha256_file(video),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            wall_time_ns=elapsed_ns,
            timed_out=timed_out,
            observation=observation,
            normalizer="ea-iris-source-frame-adapter-d969-v1",
            observed_environment={
                "parent_precondition": execution["observation"],
                "child_probe": child_probe,
            },
        )
    receipt = {
        "schema": EA_IRIS_SOURCE_RUN_SCHEMA,
        "lane": execution_lane,
        "comparator": {
            "name": EA_IRIS_SOURCE_ADAPTER_ID,
            "repository_url": build["source"]["repository_url"],
            "revision": build["source"]["revision"],
            "tree": build["source"]["tree"],
            "license": "BSD-3-Clause",
            "distribution": build["distribution"],
            "binary": str(adapter_binary),
            "binary_sha256": _sha256_file(adapter_binary),
            "working_directory": str(root),
        },
        "input": {
            "path": str(video),
            "sha256": _sha256_file(video),
            "renderer_rgb_sha256": contract["renderer_source"]["rgb_sha256"],
        },
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
        "direct_rgb_input": {
            "raw_rgb": {**direct_rgb["raw_rgb"], "path": raw_rgb.name},
            "timeline": {**direct_rgb["timeline"], "path": timeline.name},
            "timeline_payload_sha256": _canonical_json_sha256(direct_rgb["payload"]),
            "materialization_timing": "EXCLUDED_FROM_MEASURED_CHILD",
        },
        "execution_configuration": {
            "path": configuration.relative_to(root).as_posix(),
            "sha256": _sha256_file(configuration),
            "private_staged_copy": True,
        },
        "census_receipt": {"path": str(census_path), "sha256": _sha256_file(census_path)},
        "build_receipt": {"path": str(build_path), "sha256": _sha256_file(build_path)},
        "conformance_receipt": conformance_binding,
        "command": command,
        "tool_command": tool_command,
        "exit_code": completed.returncode,
        "wall_time_ns": elapsed_ns,
        "stdout": {"path": stdout_path.name, "sha256": _sha256_file(stdout_path)},
        "stderr": {"path": stderr_path.name, "sha256": _sha256_file(stderr_path)},
        "raw_output": {
            "path": raw_output.name,
            "exists": raw_output.is_file(),
            "sha256": _sha256_file(raw_output) if raw_output.is_file() else None,
            "ledger_sha256": _canonical_json_sha256(raw_payload) if raw_payload is not None else None,
        },
        "parsed_observation": observation,
        "parse_error": parse_error,
        "fair_runtime": runtime_receipt,
        "runtime_probe": {
            "path": probe_path.name,
            "sha256": _sha256_file(probe_path) if probe_path.is_file() else None,
            "observation": child_probe,
        },
        "execution_witness": "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT",
        "status": "PROCESS_VALID" if process_valid else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": (
            ["source_adapter_execution_or_parser_inconclusive"]
            if not process_valid
            else ([
                "decoder_thread_control_unsupported_not_verified",
                "decoder_prematerialization_excluded_from_runtime_boundary",
                "local_execution_witness_not_independent",
                "independent_gold_receipt_missing",
                "frozen_public_case_ledger_missing",
            ] if execution_lane == "DIRECT_DETECTOR" else [
                "conformance_only_not_direct_detector_participant",
                "decoder_thread_control_unsupported_not_verified",
                "decoder_prematerialization_excluded_from_runtime_boundary",
                "local_execution_witness_not_independent",
            ])
        ),
    }
    receipt_path = root / "ea-iris-source-adapter-run-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def _parse_iris_frame_csv_detailed(
    csv_path: Path,
    *,
    expected_frame_count: int,
) -> dict[str, object]:
    try:
        raw = csv_path.read_bytes().replace(b"\x00", b"")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ExternalLeagueError("EA IRIS source video FrameData CSV is unreadable") from exc
    required = {
        "Frame", "TimeStamp", "LuminanceFrameResult", "RedFrameResult", "PatternFrameResult",
    }
    if not rows or len(rows) != expected_frame_count or set(rows[0]) < required:
        raise ExternalLeagueError("EA IRIS source video FrameData CSV count or columns are invalid")
    categories: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            frame = int(row["Frame"])
            luminance = int(row["LuminanceFrameResult"])
            red = int(row["RedFrameResult"])
            pattern = int(row["PatternFrameResult"])
            timestamp = row["TimeStamp"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalLeagueError("EA IRIS source video FrameData categorical row is invalid") from exc
        if (
            frame != index + 1
            or luminance not in {0, 1, 2, 3}
            or red not in {0, 1, 2, 3}
            or pattern not in {0, 1}
            or not isinstance(timestamp, str)
            or not timestamp
        ):
            raise ExternalLeagueError("EA IRIS source video FrameData values are invalid")
        categories.append({
            "frame_index": index,
            "iris_frame_number": frame,
            "iris_timestamp_string": timestamp,
            "luminance_result": luminance,
            "red_result": red,
            "pattern_result": pattern,
        })
    hazardous = [
        row["frame_index"] for row in categories
        if row["luminance_result"] >= 2 or row["red_result"] >= 2 or row["pattern_result"] >= 1
    ]
    warnings = [
        row["frame_index"] for row in categories
        if row["luminance_result"] == 1 or row["red_result"] == 1
    ]
    return {
        "categories": categories,
        "categories_sha256": _canonical_json_sha256(categories),
        "prediction": "HAZARDOUS" if hazardous else "SAFE",
        "hazard_frame_indices": hazardous,
        "warning": bool(warnings),
        "warning_frame_indices": warnings,
    }


def execute_iris_source_video_conformance(
    build_receipt: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    repeat_ordinal: int,
) -> dict[str, object]:
    """Run the unmodified d969 example video path as a conformance-only oracle."""
    build, build_path = _load_iris_source_build_receipt(build_receipt)
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS source video conformance output already exists: {root}")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if contract["fps"] != 60:
        raise ExternalLeagueError("EA IRIS source video conformance accepts only canonical 60fps CFR")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen_runtime is None:
        raise ExternalLeagueError("EA IRIS source video conformance requires a frozen runtime protocol")
    if isinstance(repeat_ordinal, bool) or repeat_ordinal not in {1, 2, 3}:
        raise ExternalLeagueError("EA IRIS source video conformance repeat ordinal must be one through three")
    root.mkdir(parents=True)
    results = root / "Results" / video.name
    results.mkdir(parents=True)
    configuration = root / "appsettings.json"
    build_configuration = build_path.parent / str(build["configuration"]["path"])
    shutil.copy2(build_configuration, configuration)
    binary = _iris_built_binary(build_path, build, "source_video_oracle")
    tool_command = [str(binary), "-v", str(video), "-j", "0", "-p", "1", "-r", "0"]
    with _fair_execution_context(frozen_runtime, video, base_environment=os.environ) as execution:
        probe_path = root / "runtime-probe.json"
        command = _instrument_fair_command(execution, tool_command, probe_path)
        started = time.monotonic_ns()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=execution["environment"],
                capture_output=True,
                check=False,
                timeout=int(frozen_runtime["budget"]["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: EA IRIS d969 source video timeout",
            )
        stdout_path = root / "stdout.bin"
        stderr_path = root / "stderr.bin"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        csv_path = results / "framedata.csv"
        detailed: dict[str, object] | None = None
        parse_error: str | None = None
        if completed.returncode == 0 and csv_path.is_file():
            try:
                detailed = _parse_iris_frame_csv_detailed(
                    csv_path,
                    expected_frame_count=int(contract["frame_count"]),
                )
            except ExternalLeagueError as exc:
                parse_error = str(exc)
        child_probe: dict[str, object] | None = None
        try:
            child_probe = _load_child_runtime_probe(probe_path)
        except ExternalLeagueError:
            pass
        finished = time.monotonic_ns()
        elapsed_ns = finished - started
        if elapsed_ns > int(frozen_runtime["budget"]["timeout_seconds"]) * 1_000_000_000:
            timed_out = True
        process_valid = (
            completed.returncode == 0
            and not timed_out
            and detailed is not None
            and child_probe is not None
        )
        observation = None if detailed is None else {
            "tool": EA_IRIS_SOURCE_VIDEO_ORACLE_ID,
            "distribution": "unmodified-source-example-path-d96978ac-conformance-only",
            "prediction": detailed["prediction"],
            "warning": detailed["warning"],
            "frame_count": contract["frame_count"],
            "fps": 60,
            "hazard_frame_indices": detailed["hazard_frame_indices"],
            "warning_frame_indices": detailed["warning_frame_indices"],
            "timestamp_metrics": "native-rounded-csv-conformance-only",
            "mask_metrics": "NOT_APPLICABLE",
        }
        runtime_receipt = _fair_runtime_run_receipt(
            frozen_runtime,
            comparator=EA_IRIS_SOURCE_VIDEO_ORACLE_ID,
            scheduled_repeat_ordinal=repeat_ordinal,
            schedule_binding=None,
            input_sha256=_sha256_file(video),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            wall_time_ns=elapsed_ns,
            timed_out=timed_out,
            observation=observation,
            normalizer="ea-iris-d969-unmodified-source-video-csv-v1",
            observed_environment={
                "parent_precondition": execution["observation"],
                "child_probe": child_probe,
            },
        )
    receipt = {
        "schema": EA_IRIS_SOURCE_VIDEO_RUN_SCHEMA,
        "identity": EA_IRIS_SOURCE_VIDEO_ORACLE_ID,
        "lane": "CONFORMANCE_ONLY",
        "source_revision": EA_IRIS_SOURCE_REVISION,
        "unmodified_example_source_sha256": EA_IRIS_SOURCE_EXAMPLE_SHA256,
        "build_receipt": {"path": str(build_path), "sha256": _sha256_file(build_path)},
        "input": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
        "configuration": {"path": configuration.name, "sha256": _sha256_file(configuration)},
        "command": command,
        "tool_command": tool_command,
        "exit_code": completed.returncode,
        "stdout": {"path": stdout_path.name, "sha256": _sha256_file(stdout_path)},
        "stderr": {"path": stderr_path.name, "sha256": _sha256_file(stderr_path)},
        "frame_report": {
            "path": str(csv_path.relative_to(root)),
            "exists": csv_path.is_file(),
            "sha256": _sha256_file(csv_path) if csv_path.is_file() else None,
            "categories_sha256": detailed.get("categories_sha256") if detailed is not None else None,
        },
        "parsed_observation": observation,
        "parse_error": parse_error,
        "fair_runtime": runtime_receipt,
        "runtime_probe": {
            "path": probe_path.name,
            "sha256": _sha256_file(probe_path) if probe_path.is_file() else None,
            "observation": child_probe,
        },
        "status": "PROCESS_VALID" if process_valid else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": [
            "conformance_only_not_direct_detector_participant",
            "local_execution_witness_not_independent",
        ],
    }
    receipt_path = root / "ea-iris-source-video-conformance-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def execute_iris_realtime_semantic_probe(
    build_receipt: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
) -> dict[str, object]:
    """Record the public real-time API's decisive pattern-timing mismatch.

    This probe is deliberately non-scoring.  It executes the direct
    ``RealTimeInit`` lane and the unmodified d969 ``example/main.cpp`` video
    lane on the same 31-frame upstream pattern image.  Terminal agreement is
    insufficient: any frame-category/timing difference keeps the direct lane
    NOT_VERIFIED.
    """
    build, build_path = _load_iris_source_build_receipt(build_receipt)
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS semantic probe output already exists: {root}")
    decoded_rgb, _, _ = _decode_canonical_video_rgb(video, conversion)
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if (
        contract["fps"] != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["fps"]
        or contract["frame_count"] != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["frame_count"]
        or list(decoded_rgb.shape) != [31, 360, 640, 3]
        or any(
            _sha256_bytes(np.ascontiguousarray(frame).tobytes())
            != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["expected_rgb_frame_sha256"]
            for frame in decoded_rgb
        )
    ):
        raise ExternalLeagueError("EA IRIS semantic probe input is not the frozen upstream pattern fixture")
    source_root = Path(str(build["source"]["path"])).resolve()
    upstream_image = source_root / str(EA_IRIS_PATTERN_NEGATIVE_FIXTURE["path"])
    if (
        not upstream_image.is_file()
        or _sha256_file(upstream_image) != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS upstream pattern fixture bytes drifted")

    root.mkdir(parents=True)
    direct_root = root / "direct"
    direct_root.mkdir()
    direct_rgb = _materialize_iris_direct_rgb_input(video, conversion, direct_root)
    raw_rgb = Path(str(direct_rgb["raw_rgb"]["path"])).resolve()
    timeline = Path(str(direct_rgb["timeline"]["path"])).resolve()
    direct_config_root = direct_root / "configuration"
    direct_config_root.mkdir()
    direct_config = direct_config_root / "appsettings.json"
    build_config = build_path.parent / str(build["configuration"]["path"])
    shutil.copy2(build_config, direct_config)
    direct_config.chmod(0o444)
    direct_output = direct_root / "adapter-output.json"
    direct_binary = _iris_built_binary(build_path, build, "source_frame_adapter")
    direct_command = [
        str(direct_binary),
        str(raw_rgb),
        str(timeline),
        str(direct_config_root) + os.sep,
        str(direct_output),
        "640",
        "360",
        "31",
        _sha256_file(raw_rgb),
        _sha256_file(timeline),
        _sha256_file(direct_config),
        "1",
    ]
    frozen_environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    direct_process = subprocess.run(
        direct_command,
        capture_output=True,
        check=False,
        timeout=120,
        env=frozen_environment,
    )
    direct_stdout = direct_root / "stdout.bin"
    direct_stderr = direct_root / "stderr.bin"
    direct_stdout.write_bytes(direct_process.stdout)
    direct_stderr.write_bytes(direct_process.stderr)
    if direct_process.returncode != 0 or not direct_output.is_file():
        raise ExternalLeagueError("EA IRIS direct semantic probe process failed")
    direct_raw, direct_observation = _parse_iris_source_adapter_output(
        direct_output,
        video=video,
        conversion=conversion,
        build=build,
        direct_rgb=direct_rgb,
        execution_configuration=direct_config,
        expected_thread_limit=1,
    )

    source_root_run = root / "source-video"
    source_root_run.mkdir()
    source_config = source_root_run / "appsettings.json"
    shutil.copy2(build_config, source_config)
    source_binary = _iris_built_binary(build_path, build, "source_video_oracle")
    source_command = [str(source_binary), "-v", str(video), "-j", "0", "-p", "1", "-r", "0"]
    source_process = subprocess.run(
        source_command,
        cwd=str(source_root_run),
        capture_output=True,
        check=False,
        timeout=120,
        env=frozen_environment,
    )
    source_stdout = source_root_run / "stdout.bin"
    source_stderr = source_root_run / "stderr.bin"
    source_stdout.write_bytes(source_process.stdout)
    source_stderr.write_bytes(source_process.stderr)
    frame_report = source_root_run / "Results" / video.name / "framedata.csv"
    if source_process.returncode != 0 or not frame_report.is_file():
        raise ExternalLeagueError("EA IRIS source-video semantic probe process failed")
    source_detailed = _parse_iris_frame_csv_detailed(frame_report, expected_frame_count=31)
    direct_categories = _iris_category_projection([
        {
            "frame_index": row["frame_index"],
            "luminance_result": row["native_frame_data"]["luminance_result"]["code"],
            "red_result": row["native_frame_data"]["red_result"]["code"],
            "pattern_result": row["native_frame_data"]["pattern_result"]["code"],
        }
        for row in direct_raw["frames"]
    ])
    source_categories = _iris_category_projection(source_detailed["categories"])
    parity = _iris_semantic_parity_evidence(
        direct_categories,
        source_categories,
        direct_prediction=str(direct_observation["prediction"]),
        direct_warning=bool(direct_observation["warning"]),
        source_prediction=str(source_detailed["prediction"]),
        source_warning=bool(source_detailed["warning"]),
    )
    direct_pattern_frames = parity["direct_pattern_fail_frames"]
    source_pattern_frames = parity["source_pattern_fail_frames"]
    comparison = parity["comparison"]
    if (
        not direct_pattern_frames
        or not source_pattern_frames
        or parity["verified"] is not False
        or comparison["terminal_agreement"] is not True
    ):
        raise ExternalLeagueError("EA IRIS mandatory semantic mismatch was not reproduced exactly")

    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    receipt = {
        "schema": EA_IRIS_REALTIME_SEMANTIC_PROBE_SCHEMA,
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "classification": "MANDATORY_NEGATIVE_CONFORMANCE_NOT_SCORING",
        "source_build": {
            "path": str(build_path),
            "sha256": _sha256_file(build_path),
            "direct_binary_sha256": build["binaries"]["source_frame_adapter"]["sha256"],
            "source_video_binary_sha256": build["binaries"]["source_video_oracle"]["sha256"],
        },
        "input": {
            "canonical_video": {"path": str(video), "sha256": _sha256_file(video)},
            "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
            "upstream_pattern_fixture": {
                **EA_IRIS_PATTERN_NEGATIVE_FIXTURE,
                "source_path": str(upstream_image),
            },
            "frame_map_sha256": contract["frame_map_sha256"],
        },
        "direct": {
            "command": direct_command,
            "exit_code": direct_process.returncode,
            "raw_rgb": artifact(raw_rgb),
            "timeline": artifact(timeline),
            "configuration": artifact(direct_config),
            "output": artifact(direct_output),
            "stdout": artifact(direct_stdout),
            "stderr": artifact(direct_stderr),
            "categories_sha256": _canonical_json_sha256(direct_categories),
            "pattern_fail_frames": direct_pattern_frames,
            "terminal": {
                "prediction": direct_observation["prediction"],
                "warning": direct_observation["warning"],
            },
        },
        "source_video": {
            "command": source_command,
            "exit_code": source_process.returncode,
            "configuration": artifact(source_config),
            "frame_report": artifact(frame_report),
            "stdout": artifact(source_stdout),
            "stderr": artifact(source_stderr),
            "categories_sha256": _canonical_json_sha256(source_categories),
            "pattern_fail_frames": source_pattern_frames,
            "terminal": {
                "prediction": source_detailed["prediction"],
                "warning": source_detailed["warning"],
            },
        },
        "comparison": comparison,
        "status": "SEMANTIC_MISMATCH_NOT_VERIFIED",
        "direct_participant_authorized": False,
        "scoreable": False,
        "scoreable_blockers": [
            "realtime_and_source_video_frame_category_timing_mismatch",
            "terminal_agreement_is_insufficient",
            "release_equivalence_not_claimed",
        ],
    }
    receipt_path = root / "ea-iris-realtime-semantic-probe.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": _sha256_file(receipt_path)}


def freeze_iris_source_conformance_manifest(
    fixtures: Sequence[Mapping[str, object]],
    build_receipt: Path | str,
    release_census_receipt: Path | str,
    destination: Path | str,
) -> dict[str, object]:
    """Freeze a non-scoring fixture corpus before opening any detector output."""
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"EA IRIS conformance manifest already exists: {output}")
    build, build_path = _load_iris_source_build_receipt(build_receipt)
    release_census = Path(release_census_receipt).resolve()
    if not release_census.is_file():
        raise ExternalLeagueError("EA IRIS release census receipt is unavailable")
    try:
        release_census_payload = json.loads(release_census.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS release census receipt is unreadable") from exc
    artifact_root = release_census_payload.get("artifact_root") if isinstance(release_census_payload, Mapping) else None
    if release_census_payload.get("schema") != COMPARATOR_CENSUS_RECEIPT_SCHEMA or not isinstance(artifact_root, str):
        raise ExternalLeagueError("EA IRIS release census receipt schema is invalid")
    release_entry, _ = _load_execution_census_entry(
        release_census,
        artifact_root,
        EA_IRIS_RELEASE_ORACLE_ID,
    )
    required_boundary_pairs = {
        (family, value)
        for family, values in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES.items()
        for value in values
    }
    if (
        not isinstance(fixtures, Sequence)
        or isinstance(fixtures, (str, bytes))
        or len(fixtures) < len(required_boundary_pairs)
    ):
        raise ExternalLeagueError("EA IRIS conformance corpus omits mandatory temporal boundary fixtures")
    rows: list[dict[str, object]] = []
    identities: set[str] = set()
    observed_boundaries: set[tuple[str, int]] = set()
    for raw in fixtures:
        if not isinstance(raw, Mapping) or set(raw) != {
            "fixture_id", "coverage_role", "temporal_boundary",
            "canonical_video", "conversion_receipt",
        }:
            raise ExternalLeagueError("EA IRIS conformance fixture fields are invalid")
        fixture_id = raw.get("fixture_id")
        coverage_role = raw.get("coverage_role")
        if (
            not isinstance(fixture_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", fixture_id)
            or fixture_id in identities
            or coverage_role not in set(EA_IRIS_REQUIRED_CONFORMANCE_ROLES)
        ):
            raise ExternalLeagueError("EA IRIS conformance fixture identity or role is invalid")
        identities.add(fixture_id)
        video = Path(str(raw["canonical_video"])).resolve()
        conversion = Path(str(raw["conversion_receipt"])).resolve()
        _, contract = _canonical_decoder_timeline_contract(video, conversion)
        if contract["fps"] != 60:
            raise ExternalLeagueError("EA IRIS conformance corpus accepts only 60fps CFR fixtures")
        boundary = raw.get("temporal_boundary")
        normalized_boundary: dict[str, object] | None = None
        if boundary is not None:
            if not isinstance(boundary, Mapping) or set(boundary) != {"family", "value", "unit"}:
                raise ExternalLeagueError("EA IRIS temporal boundary fixture contract is invalid")
            family = boundary.get("family")
            value = boundary.get("value")
            expected_unit = "count" if family == "TRANSITION_COUNT" else "frames"
            if (
                not isinstance(family, str)
                or family not in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value not in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES[family]
                or boundary.get("unit") != expected_unit
                or (expected_unit == "frames" and contract["frame_count"] != value)
                or (family, value) in observed_boundaries
            ):
                raise ExternalLeagueError("EA IRIS temporal boundary identity, unit, or frame count drifted")
            observed_boundaries.add((family, value))
            normalized_boundary = {"family": family, "value": value, "unit": expected_unit}
        rows.append({
            "fixture_id": fixture_id,
            "fixture_class": "CONFORMANCE_ONLY",
            "coverage_role": coverage_role,
            "temporal_boundary": normalized_boundary,
            "scoring_eligible": False,
            "canonical_video": contract["canonical_video"],
            "conversion_receipt": contract["conversion_receipt"],
            "renderer_source": contract["renderer_source"],
            "frame_count": contract["frame_count"],
            "fps": contract["fps"],
            "frame_map_sha256": contract["frame_map_sha256"],
        })
    if {row["coverage_role"] for row in rows} != set(EA_IRIS_REQUIRED_CONFORMANCE_ROLES):
        raise ExternalLeagueError("EA IRIS conformance corpus must cover all four mandatory roles")
    if observed_boundaries != required_boundary_pairs:
        raise ExternalLeagueError("EA IRIS conformance corpus does not cover every mandatory temporal boundary")
    for identity_field in ("canonical_video", "renderer_source"):
        hashes = [str(row[identity_field]["sha256"]) for row in rows]
        if len(set(hashes)) != len(hashes):
            raise ExternalLeagueError("EA IRIS conformance fixtures must use distinct canonical inputs")
    frame_map_hashes = [str(row["frame_map_sha256"]) for row in rows]
    if len(set(frame_map_hashes)) != len(frame_map_hashes):
        raise ExternalLeagueError("EA IRIS conformance fixtures must use distinct frame maps")
    rows.sort(key=lambda row: str(row["fixture_id"]))
    manifest = {
        "schema": EA_IRIS_SOURCE_CONFORMANCE_MANIFEST_SCHEMA,
        "freeze_state": "PRE_EXECUTION_FROZEN",
        "corpus_class": "CONFORMANCE_ONLY_NEVER_SCORING",
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "source_build": {"path": str(build_path), "sha256": _sha256_file(build_path), "binary_sha256": build["binaries"]["source_frame_adapter"]["sha256"]},
        "source_video_oracle": {"identity": EA_IRIS_SOURCE_VIDEO_ORACLE_ID, "binary_sha256": build["binaries"]["source_video_oracle"]["sha256"]},
        "release_oracle": {
            "identity": EA_IRIS_RELEASE_ORACLE_ID,
            "census_receipt": str(release_census),
            "census_receipt_sha256": _sha256_file(release_census),
            "artifact_root": str(Path(artifact_root).resolve()),
            "provenance_sha256": release_entry["provenance_sha256"],
        },
        "fixtures": rows,
        "fixtures_sha256": _canonical_json_sha256(rows),
        "required_roles": list(EA_IRIS_REQUIRED_CONFORMANCE_ROLES),
        "required_temporal_boundaries": {
            family: list(values) for family, values in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES.items()
        },
        "release_equivalence": "NOT_CLAIMED",
        "scoreable": False,
    }
    _write_json(output, manifest)
    return {**manifest, "manifest": str(output), "manifest_sha256": _sha256_file(output)}


def _load_iris_source_adapter_run(
    receipt_ref: Path | str,
    *,
    expected_lane: str | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source adapter run receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EA_IRIS_SOURCE_RUN_SCHEMA
        or payload.get("lane") not in {"DIRECT_DETECTOR", "CONFORMANCE_ONLY"}
        or (expected_lane is not None and payload.get("lane") != expected_lane)
        or payload.get("status") != "PROCESS_VALID"
        or payload.get("scoreable") is not False
        or payload.get("execution_witness") != "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT"
    ):
        raise ExternalLeagueError("EA IRIS source adapter run receipt is not a valid local process receipt")
    build_ref = payload.get("build_receipt")
    conformance_ref = payload.get("conformance_receipt")
    conversion_ref = payload.get("conversion_receipt")
    input_ref = payload.get("input")
    raw_ref = payload.get("raw_output")
    direct_ref = payload.get("direct_rgb_input")
    execution_config_ref = payload.get("execution_configuration")
    if not all(
        isinstance(item, Mapping)
        for item in (build_ref, conversion_ref, input_ref, raw_ref, direct_ref, execution_config_ref)
    ):
        raise ExternalLeagueError("EA IRIS source adapter run artifact references are invalid")
    build_path = Path(str(build_ref.get("path", ""))).resolve()
    conversion = Path(str(conversion_ref.get("path", ""))).resolve()
    video = Path(str(input_ref.get("path", ""))).resolve()
    if (
        not build_path.is_file()
        or build_ref.get("sha256") != _sha256_file(build_path)
        or not conversion.is_file()
        or conversion_ref.get("sha256") != _sha256_file(conversion)
        or not video.is_file()
        or input_ref.get("sha256") != _sha256_file(video)
    ):
        raise ExternalLeagueError("EA IRIS source adapter run input, conversion, or build hash drifted")
    build, build_path = _load_iris_source_build_receipt(build_path)
    if payload["lane"] == "DIRECT_DETECTOR":
        if not isinstance(conformance_ref, Mapping) or not isinstance(conformance_ref.get("path"), str):
            raise ExternalLeagueError("EA IRIS direct detector run omits conformance authority")
        conformance, conformance_path = _load_iris_source_conformance_receipt(conformance_ref["path"])
        if (
            conformance_ref.get("sha256") != _sha256_file(conformance_path)
            or conformance.get("source_build") != {
                "path": str(build_path),
                "sha256": _sha256_file(build_path),
                "binary_sha256": build["binaries"]["source_frame_adapter"]["sha256"],
            }
        ):
            raise ExternalLeagueError("EA IRIS direct detector conformance authority differs from build")
    elif conformance_ref is not None:
        raise ExternalLeagueError("EA IRIS conformance-only run cannot bind a prior conformance claim")
    comparator = payload.get("comparator")
    adapter_binary = _iris_built_binary(build_path, build, "source_frame_adapter")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("name") != EA_IRIS_SOURCE_ADAPTER_ID
        or comparator.get("repository_url") != build["source"]["repository_url"]
        or comparator.get("revision") != EA_IRIS_SOURCE_REVISION
        or comparator.get("tree") != EA_IRIS_SOURCE_TREE
        or comparator.get("license") != "BSD-3-Clause"
        or comparator.get("distribution") != "local-source-build-d96978ac-not-official-release"
        or Path(str(comparator.get("binary", ""))).resolve() != adapter_binary
        or comparator.get("binary_sha256") != _sha256_file(adapter_binary)
    ):
        raise ExternalLeagueError("EA IRIS source adapter run provenance differs from source build")
    direct_raw_ref = direct_ref.get("raw_rgb")
    direct_timeline_ref = direct_ref.get("timeline")
    if not all(isinstance(item, Mapping) for item in (direct_raw_ref, direct_timeline_ref)):
        raise ExternalLeagueError("EA IRIS source adapter direct RGB references are invalid")
    for reference, label in (
        (direct_raw_ref, "EA IRIS direct raw RGB"),
        (direct_timeline_ref, "EA IRIS direct RGB timeline"),
        (execution_config_ref, "EA IRIS private execution configuration"),
    ):
        relative = reference.get("path")
        if isinstance(relative, str) and (receipt_path.parent / relative).is_symlink():
            raise ExternalLeagueError(f"{label} cannot be a symlink")
    direct_raw_path = _resolve_run_owned_artifact(
        receipt_path, direct_raw_ref.get("path"), label="EA IRIS direct raw RGB",
    )
    direct_timeline_path = _resolve_run_owned_artifact(
        receipt_path, direct_timeline_ref.get("path"), label="EA IRIS direct RGB timeline",
    )
    execution_configuration = _resolve_run_owned_artifact(
        receipt_path,
        execution_config_ref.get("path"),
        label="EA IRIS private execution configuration",
    )
    if (
        not direct_raw_path.is_file()
        or direct_raw_ref.get("sha256") != _sha256_file(direct_raw_path)
        or direct_raw_ref.get("bytes") != direct_raw_path.stat().st_size
        or not direct_timeline_path.is_file()
        or direct_timeline_ref.get("sha256") != _sha256_file(direct_timeline_path)
        or not execution_configuration.is_file()
        or execution_config_ref.get("sha256") != _sha256_file(execution_configuration)
        or execution_config_ref.get("sha256") != build["configuration"]["sha256"]
        or execution_config_ref.get("private_staged_copy") is not True
    ):
        raise ExternalLeagueError("EA IRIS direct RGB or private configuration artifact drifted")
    direct_payload = _load_iris_direct_rgb_input(
        direct_raw_path,
        direct_timeline_path,
        video=video,
        conversion=conversion,
        expected_raw_sha256=str(direct_raw_ref["sha256"]),
        expected_timeline_sha256=str(direct_timeline_ref["sha256"]),
    )
    if (
        direct_ref.get("timeline_payload_sha256") != _canonical_json_sha256(direct_payload)
        or direct_ref.get("materialization_timing") != "EXCLUDED_FROM_MEASURED_CHILD"
    ):
        raise ExternalLeagueError("EA IRIS direct RGB timeline receipt binding drifted")
    direct_rgb = {
        "raw_rgb": {
            "path": str(direct_raw_path),
            "sha256": _sha256_file(direct_raw_path),
            "bytes": direct_raw_path.stat().st_size,
        },
        "timeline": {
            "path": str(direct_timeline_path),
            "sha256": _sha256_file(direct_timeline_path),
        },
        "payload": direct_payload,
    }
    raw_path = _resolve_run_owned_artifact(
        receipt_path,
        raw_ref.get("path"),
        label="EA IRIS source adapter raw ledger",
    )
    if (
        not raw_path.is_file()
        or raw_ref.get("exists") is not True
        or raw_ref.get("sha256") != _sha256_file(raw_path)
    ):
        raise ExternalLeagueError("EA IRIS source adapter raw ledger hash drifted")
    tool_command = payload.get("tool_command")
    if (
        not isinstance(tool_command, list)
        or len(tool_command) != 12
        or not isinstance(tool_command[-1], str)
        or not tool_command[-1].isdigit()
        or int(tool_command[-1]) <= 0
    ):
        raise ExternalLeagueError("EA IRIS source adapter tool command or thread limit is invalid")
    expected_thread_limit = int(tool_command[-1])
    raw, observation = _parse_iris_source_adapter_output(
        raw_path,
        video=video,
        conversion=conversion,
        build=build,
        direct_rgb=direct_rgb,
        execution_configuration=execution_configuration,
        expected_thread_limit=expected_thread_limit,
    )
    if raw_ref.get("ledger_sha256") != _canonical_json_sha256(raw):
        raise ExternalLeagueError("EA IRIS source adapter raw ledger canonical hash mismatches")
    if payload.get("parsed_observation") != observation or payload.get("parse_error") is not None:
        raise ExternalLeagueError("EA IRIS source adapter stored observation differs from raw ledger")
    _, height, width, channels = _canonical_decoder_timeline_contract(video, conversion)[1]["shape"]
    if channels != 3:
        raise ExternalLeagueError("EA IRIS source adapter direct RGB channel count drifted")
    expected_tool_command = [
        str(adapter_binary),
        str(direct_raw_path),
        str(direct_timeline_path),
        str(execution_configuration.parent) + os.sep,
        str(raw_path),
        str(width),
        str(height),
        str(observation["frame_count"]),
        _sha256_file(direct_raw_path),
        _sha256_file(direct_timeline_path),
        _sha256_file(execution_configuration),
        str(expected_thread_limit),
    ]
    if tool_command != expected_tool_command or payload.get("exit_code") != 0:
        raise ExternalLeagueError("EA IRIS source adapter run command or exit status drifted")
    for field in ("stdout", "stderr", "runtime_probe"):
        reference = payload.get(field)
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ExternalLeagueError("EA IRIS source adapter run log or probe reference is invalid")
        artifact = _resolve_run_owned_artifact(receipt_path, reference["path"], label=f"EA IRIS source adapter {field}")
        if not artifact.is_file() or reference.get("sha256") != _sha256_file(artifact):
            raise ExternalLeagueError("EA IRIS source adapter run log or probe hash drifted")
    probe_observation = payload["runtime_probe"].get("observation")
    resource_environment = (
        probe_observation.get("resource_environment")
        if isinstance(probe_observation, Mapping)
        else None
    )
    if (
        not isinstance(resource_environment, Mapping)
        or resource_environment.get("OMP_NUM_THREADS") != str(expected_thread_limit)
    ):
        raise ExternalLeagueError("EA IRIS source adapter OpenCV thread limit differs from child runtime policy")
    return dict(payload), raw, observation, receipt_path


def _load_iris_source_video_run(
    receipt_ref: Path | str,
) -> tuple[dict[str, object], dict[str, object], Path]:
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source video conformance receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EA_IRIS_SOURCE_VIDEO_RUN_SCHEMA
        or payload.get("identity") != EA_IRIS_SOURCE_VIDEO_ORACLE_ID
        or payload.get("lane") != "CONFORMANCE_ONLY"
        or payload.get("source_revision") != EA_IRIS_SOURCE_REVISION
        or payload.get("unmodified_example_source_sha256") != EA_IRIS_SOURCE_EXAMPLE_SHA256
        or payload.get("status") != "PROCESS_VALID"
        or payload.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS source video conformance identity or status is invalid")
    build_ref = payload.get("build_receipt")
    input_ref = payload.get("input")
    conversion_ref = payload.get("conversion_receipt")
    if not all(isinstance(item, Mapping) for item in (build_ref, input_ref, conversion_ref)):
        raise ExternalLeagueError("EA IRIS source video conformance artifact references are invalid")
    build_path = Path(str(build_ref.get("path", ""))).resolve()
    video = Path(str(input_ref.get("path", ""))).resolve()
    conversion = Path(str(conversion_ref.get("path", ""))).resolve()
    if (
        not build_path.is_file()
        or build_ref.get("sha256") != _sha256_file(build_path)
        or not video.is_file()
        or input_ref.get("sha256") != _sha256_file(video)
        or not conversion.is_file()
        or conversion_ref.get("sha256") != _sha256_file(conversion)
    ):
        raise ExternalLeagueError("EA IRIS source video conformance build or input drifted")
    build, build_path = _load_iris_source_build_receipt(build_path)
    binary = _iris_built_binary(build_path, build, "source_video_oracle")
    expected_tool_command = [str(binary), "-v", str(video), "-j", "0", "-p", "1", "-r", "0"]
    if payload.get("tool_command") != expected_tool_command or payload.get("exit_code") != 0:
        raise ExternalLeagueError("EA IRIS source video conformance command or exit status drifted")
    frame_ref = payload.get("frame_report")
    if not isinstance(frame_ref, Mapping) or not isinstance(frame_ref.get("path"), str):
        raise ExternalLeagueError("EA IRIS source video conformance FrameData report is missing")
    frame_path = _resolve_run_owned_artifact(receipt_path, frame_ref["path"], label="EA IRIS source video FrameData report")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if not frame_path.is_file() or frame_ref.get("sha256") != _sha256_file(frame_path):
        raise ExternalLeagueError("EA IRIS source video conformance FrameData hash drifted")
    detailed = _parse_iris_frame_csv_detailed(frame_path, expected_frame_count=int(contract["frame_count"]))
    if frame_ref.get("categories_sha256") != detailed["categories_sha256"]:
        raise ExternalLeagueError("EA IRIS source video categorical ledger hash mismatches")
    expected_observation = {
        "tool": EA_IRIS_SOURCE_VIDEO_ORACLE_ID,
        "distribution": "unmodified-source-example-path-d96978ac-conformance-only",
        "prediction": detailed["prediction"],
        "warning": detailed["warning"],
        "frame_count": contract["frame_count"],
        "fps": 60,
        "hazard_frame_indices": detailed["hazard_frame_indices"],
        "warning_frame_indices": detailed["warning_frame_indices"],
        "timestamp_metrics": "native-rounded-csv-conformance-only",
        "mask_metrics": "NOT_APPLICABLE",
    }
    if payload.get("parsed_observation") != expected_observation or payload.get("parse_error") is not None:
        raise ExternalLeagueError("EA IRIS source video stored observation differs from FrameData")
    return dict(payload), detailed, receipt_path


def _load_iris_release_conformance_run(
    receipt_ref: Path | str,
    *,
    video: Path,
    conversion: Path,
    census_entry: Mapping[str, object],
    census_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS official release conformance receipt is unreadable") from exc
    comparator = payload.get("comparator") if isinstance(payload, Mapping) else None
    release_asset = comparator.get("release_asset") if isinstance(comparator, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "flashpatch-ea-iris-release-run-v1"
        or payload.get("status") != "PROCESS_VALID"
        or payload.get("scoreable") is not False
        or not isinstance(comparator, Mapping)
        or comparator.get("name") != EA_IRIS_RELEASE_ORACLE_ID
        or comparator.get("repository_url") != census_entry.get("repository_url")
        or comparator.get("source_revision") != census_entry.get("distribution_source_revision")
        or comparator.get("release_tag") != census_entry.get("distribution_revision")
        or not isinstance(release_asset, Mapping)
        or release_asset.get("sha256") != census_entry.get("release_asset_sha256")
        or comparator.get("executable_sha256") != census_entry.get("binary_sha256")
        or comparator.get("appsettings_sha256") != census_entry.get("configuration_sha256")
    ):
        raise ExternalLeagueError("EA IRIS official release conformance identity or status is invalid")
    release_asset_path = Path(str(release_asset.get("path", ""))).resolve()
    if not release_asset_path.is_file() or release_asset.get("sha256") != _sha256_file(release_asset_path):
        raise ExternalLeagueError("EA IRIS official release asset drifted")
    census_ref = payload.get("census_receipt")
    if census_ref != {"path": str(census_path), "sha256": _sha256_file(census_path)}:
        raise ExternalLeagueError("EA IRIS official release conformance census binding drifted")
    conversion_ref = payload.get("conversion_receipt")
    input_ref = payload.get("input")
    if (
        not isinstance(conversion_ref, Mapping)
        or Path(str(conversion_ref.get("path", ""))).resolve() != conversion
        or conversion_ref.get("sha256") != _sha256_file(conversion)
        or not isinstance(input_ref, Mapping)
        or input_ref.get("canonical_video_sha256") != _sha256_file(video)
    ):
        raise ExternalLeagueError("EA IRIS official release conformance input differs from fixture")
    frame_ref = payload.get("frame_report")
    if not isinstance(frame_ref, Mapping) or not isinstance(frame_ref.get("path"), str):
        raise ExternalLeagueError("EA IRIS official release conformance frame report is missing")
    frame_path = _resolve_run_owned_artifact(receipt_path, frame_ref["path"], label="EA IRIS release frame report")
    stdout_path = receipt_path.parent / "stdout.bin"
    stderr_path = receipt_path.parent / "stderr.bin"
    staged_binary = receipt_path.parent / "IrisApp"
    staged_settings = receipt_path.parent / "appsettings.json"
    staged_video = receipt_path.parent / "TestVideos" / "canonical.ffv1.mkv"
    expected_tool_command = [
        str(staged_binary), "-j", str(staged_video.relative_to(receipt_path.parent)),
        "-p", "1", "-r", "0",
    ]
    command = payload.get("command")
    if (
        not frame_path.is_file()
        or frame_ref.get("sha256") != _sha256_file(frame_path)
        or not stdout_path.is_file()
        or payload.get("stdout_sha256") != _sha256_file(stdout_path)
        or not stderr_path.is_file()
        or payload.get("stderr_sha256") != _sha256_file(stderr_path)
        or not staged_binary.is_file()
        or _sha256_file(staged_binary) != census_entry.get("binary_sha256")
        or not staged_settings.is_file()
        or _sha256_file(staged_settings) != census_entry.get("configuration_sha256")
        or not staged_video.is_file()
        or _sha256_file(staged_video) != _sha256_file(video)
        or payload.get("input", {}).get("sha256") != _sha256_file(staged_video)
        or payload.get("exit_code") != 0
        or not isinstance(command, list)
        or command[-len(expected_tool_command):] != expected_tool_command
    ):
        raise ExternalLeagueError("EA IRIS official release conformance raw output hash drifted")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    detailed = _parse_iris_frame_csv_detailed(frame_path, expected_frame_count=int(contract["frame_count"]))
    normalized = parse_iris_release_csv(
        frame_path,
        stdout_path,
        expected_frame_count=int(contract["frame_count"]),
        expected_fps=60,
    )
    if payload.get("parsed_observation") != normalized or payload.get("parse_error") is not None:
        raise ExternalLeagueError("EA IRIS official release stored observation differs from raw output")
    return dict(payload), detailed, receipt_path


def _iris_category_projection(rows: Sequence[Mapping[str, object]]) -> list[dict[str, int]]:
    return [
        {
            "frame_index": int(row["frame_index"]),
            "luminance_result": int(row["luminance_result"]),
            "red_result": int(row["red_result"]),
            "pattern_result": int(row["pattern_result"]),
        }
        for row in rows
    ]


def _iris_semantic_parity_evidence(
    direct_categories: Sequence[Mapping[str, int]],
    source_categories: Sequence[Mapping[str, int]],
    *,
    direct_prediction: str,
    direct_warning: bool,
    source_prediction: str,
    source_warning: bool,
) -> dict[str, object]:
    """Keep terminal agreement from concealing frame-category timing drift."""
    if (
        not direct_categories
        or len(direct_categories) != len(source_categories)
        or any(
            set(row) != {"frame_index", "luminance_result", "red_result", "pattern_result"}
            for row in (*direct_categories, *source_categories)
        )
        or [row["frame_index"] for row in direct_categories] != list(range(len(direct_categories)))
        or [row["frame_index"] for row in source_categories] != list(range(len(source_categories)))
    ):
        raise ExternalLeagueError("EA IRIS semantic parity category sequences are malformed")
    mismatch_indices = [
        index for index, (direct_row, source_row) in enumerate(zip(direct_categories, source_categories))
        if direct_row != source_row
    ]
    direct_pattern_frames = [
        int(row["frame_index"]) for row in direct_categories if int(row["pattern_result"]) >= 1
    ]
    source_pattern_frames = [
        int(row["frame_index"]) for row in source_categories if int(row["pattern_result"]) >= 1
    ]
    comparison = {
        "terminal_agreement": direct_prediction == source_prediction and direct_warning == source_warning,
        "frame_categories_exact": not mismatch_indices,
        "mismatch_frame_indices": mismatch_indices,
        "first_direct_pattern_fail_frame": direct_pattern_frames[0] if direct_pattern_frames else None,
        "first_source_video_pattern_fail_frame": source_pattern_frames[0] if source_pattern_frames else None,
        "normalization_applied": False,
    }
    return {
        "comparison": comparison,
        "direct_pattern_fail_frames": direct_pattern_frames,
        "source_pattern_fail_frames": source_pattern_frames,
        "verified": bool(comparison["terminal_agreement"] and comparison["frame_categories_exact"]),
    }


def _load_iris_realtime_semantic_probe(
    receipt_ref: Path | str,
) -> tuple[dict[str, object], Path]:
    """Freshly parse every stored artifact in the mandatory negative probe."""
    receipt_path = Path(receipt_ref).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS semantic probe receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EA_IRIS_REALTIME_SEMANTIC_PROBE_SCHEMA
        or payload.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID
        or payload.get("classification") != "MANDATORY_NEGATIVE_CONFORMANCE_NOT_SCORING"
        or payload.get("status") != "SEMANTIC_MISMATCH_NOT_VERIFIED"
        or payload.get("direct_participant_authorized") is not False
        or payload.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS semantic probe claim boundary is invalid")
    root = receipt_path.parent
    build_ref = payload.get("source_build")
    input_ref = payload.get("input")
    direct = payload.get("direct")
    source_video = payload.get("source_video")
    comparison = payload.get("comparison")
    if not all(isinstance(item, Mapping) for item in (build_ref, input_ref, direct, source_video, comparison)):
        raise ExternalLeagueError("EA IRIS semantic probe sections are invalid")
    build_path = Path(str(build_ref.get("path", ""))).resolve()
    if not build_path.is_file() or build_ref.get("sha256") != _sha256_file(build_path):
        raise ExternalLeagueError("EA IRIS semantic probe build receipt drifted")
    build, build_path = _load_iris_source_build_receipt(build_path)
    if (
        build_ref.get("direct_binary_sha256") != build["binaries"]["source_frame_adapter"]["sha256"]
        or build_ref.get("source_video_binary_sha256") != build["binaries"]["source_video_oracle"]["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS semantic probe binary binding drifted")
    canonical_ref = input_ref.get("canonical_video")
    conversion_ref = input_ref.get("conversion_receipt")
    upstream_ref = input_ref.get("upstream_pattern_fixture")
    if not all(isinstance(item, Mapping) for item in (canonical_ref, conversion_ref, upstream_ref)):
        raise ExternalLeagueError("EA IRIS semantic probe input ledger is invalid")
    video = Path(str(canonical_ref.get("path", ""))).resolve()
    conversion = Path(str(conversion_ref.get("path", ""))).resolve()
    upstream = Path(str(upstream_ref.get("source_path", ""))).resolve()
    expected_upstream = {**EA_IRIS_PATTERN_NEGATIVE_FIXTURE, "source_path": str(upstream)}
    if (
        not video.is_file()
        or canonical_ref.get("sha256") != _sha256_file(video)
        or not conversion.is_file()
        or conversion_ref.get("sha256") != _sha256_file(conversion)
        or dict(upstream_ref) != expected_upstream
        or not upstream.is_file()
        or _sha256_file(upstream) != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS semantic probe input or upstream fixture drifted")
    decoded_rgb, _, _ = _decode_canonical_video_rgb(video, conversion)
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    if (
        input_ref.get("frame_map_sha256") != contract["frame_map_sha256"]
        or list(decoded_rgb.shape) != [31, 360, 640, 3]
        or any(
            _sha256_bytes(np.ascontiguousarray(frame).tobytes())
            != EA_IRIS_PATTERN_NEGATIVE_FIXTURE["expected_rgb_frame_sha256"]
            for frame in decoded_rgb
        )
    ):
        raise ExternalLeagueError("EA IRIS semantic probe canonical frame identity drifted")

    def resolve_artifact(reference: object, label: str) -> Path:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ExternalLeagueError(f"EA IRIS semantic probe {label} reference is invalid")
        path = (root / str(reference["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError(f"EA IRIS semantic probe {label} escapes root") from exc
        if (
            not path.is_file()
            or reference.get("bytes") != path.stat().st_size
            or reference.get("sha256") != _sha256_file(path)
        ):
            raise ExternalLeagueError(f"EA IRIS semantic probe {label} bytes drifted")
        return path

    direct_raw_path = resolve_artifact(direct.get("raw_rgb"), "direct raw RGB")
    direct_timeline_path = resolve_artifact(direct.get("timeline"), "direct timeline")
    direct_config = resolve_artifact(direct.get("configuration"), "direct configuration")
    direct_output = resolve_artifact(direct.get("output"), "direct output")
    resolve_artifact(direct.get("stdout"), "direct stdout")
    resolve_artifact(direct.get("stderr"), "direct stderr")
    direct_payload = _load_iris_direct_rgb_input(
        direct_raw_path,
        direct_timeline_path,
        video=video,
        conversion=conversion,
        expected_raw_sha256=_sha256_file(direct_raw_path),
        expected_timeline_sha256=_sha256_file(direct_timeline_path),
    )
    direct_rgb = {
        "raw_rgb": {
            "path": str(direct_raw_path),
            "sha256": _sha256_file(direct_raw_path),
            "bytes": direct_raw_path.stat().st_size,
        },
        "timeline": {"path": str(direct_timeline_path), "sha256": _sha256_file(direct_timeline_path)},
        "payload": direct_payload,
    }
    direct_raw, direct_observation = _parse_iris_source_adapter_output(
        direct_output,
        video=video,
        conversion=conversion,
        build=build,
        direct_rgb=direct_rgb,
        execution_configuration=direct_config,
        expected_thread_limit=1,
    )
    direct_categories = _iris_category_projection([
        {
            "frame_index": row["frame_index"],
            "luminance_result": row["native_frame_data"]["luminance_result"]["code"],
            "red_result": row["native_frame_data"]["red_result"]["code"],
            "pattern_result": row["native_frame_data"]["pattern_result"]["code"],
        }
        for row in direct_raw["frames"]
    ])
    source_config = resolve_artifact(source_video.get("configuration"), "source-video configuration")
    frame_report = resolve_artifact(source_video.get("frame_report"), "source-video frame report")
    resolve_artifact(source_video.get("stdout"), "source-video stdout")
    resolve_artifact(source_video.get("stderr"), "source-video stderr")
    if _sha256_file(source_config) != build["configuration"]["sha256"]:
        raise ExternalLeagueError("EA IRIS semantic probe source-video configuration drifted")
    source_detailed = _parse_iris_frame_csv_detailed(frame_report, expected_frame_count=31)
    source_categories = _iris_category_projection(source_detailed["categories"])
    parity = _iris_semantic_parity_evidence(
        direct_categories,
        source_categories,
        direct_prediction=str(direct_observation["prediction"]),
        direct_warning=bool(direct_observation["warning"]),
        source_prediction=str(source_detailed["prediction"]),
        source_warning=bool(source_detailed["warning"]),
    )
    direct_pattern_frames = parity["direct_pattern_fail_frames"]
    source_pattern_frames = parity["source_pattern_fail_frames"]
    expected_comparison = parity["comparison"]
    if (
        direct.get("categories_sha256") != _canonical_json_sha256(direct_categories)
        or direct.get("pattern_fail_frames") != direct_pattern_frames
        or direct.get("terminal") != {
            "prediction": direct_observation["prediction"], "warning": direct_observation["warning"],
        }
        or source_video.get("categories_sha256") != _canonical_json_sha256(source_categories)
        or source_video.get("pattern_fail_frames") != source_pattern_frames
        or source_video.get("terminal") != {
            "prediction": source_detailed["prediction"], "warning": source_detailed["warning"],
        }
        or dict(comparison) != expected_comparison
        or not expected_comparison["terminal_agreement"]
        or parity["verified"] is not False
    ):
        raise ExternalLeagueError("EA IRIS semantic mismatch receipt differs from raw category evidence")
    direct_binary = _iris_built_binary(build_path, build, "source_frame_adapter")
    source_binary = _iris_built_binary(build_path, build, "source_video_oracle")
    expected_direct_command = [
        str(direct_binary), str(direct_raw_path), str(direct_timeline_path),
        str(direct_config.parent) + os.sep, str(direct_output), "640", "360", "31",
        _sha256_file(direct_raw_path), _sha256_file(direct_timeline_path),
        _sha256_file(direct_config), "1",
    ]
    expected_source_command = [
        str(source_binary), "-v", str(video), "-j", "0", "-p", "1", "-r", "0",
    ]
    if (
        direct.get("command") != expected_direct_command
        or direct.get("exit_code") != 0
        or source_video.get("command") != expected_source_command
        or source_video.get("exit_code") != 0
    ):
        raise ExternalLeagueError("EA IRIS semantic probe command ledger drifted")
    return dict(payload), receipt_path


def verify_iris_source_adapter_conformance(
    conformance_manifest: Path | str,
    run_sets: Sequence[Mapping[str, object]],
    destination: Path | str,
) -> dict[str, object]:
    """Produce a differential-only report across d969 API, d969 video and 1.1.0."""
    manifest_path = Path(conformance_manifest).resolve()
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"EA IRIS conformance receipt already exists: {output}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS conformance manifest is unreadable") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != EA_IRIS_SOURCE_CONFORMANCE_MANIFEST_SCHEMA
        or manifest.get("freeze_state") != "PRE_EXECUTION_FROZEN"
        or manifest.get("corpus_class") != "CONFORMANCE_ONLY_NEVER_SCORING"
        or manifest.get("identity") != EA_IRIS_SOURCE_ADAPTER_ID
        or manifest.get("release_equivalence") != "NOT_CLAIMED"
        or manifest.get("scoreable") is not False
        or manifest.get("required_roles") != list(EA_IRIS_REQUIRED_CONFORMANCE_ROLES)
        or manifest.get("required_temporal_boundaries") != {
            family: list(values) for family, values in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES.items()
        }
    ):
        raise ExternalLeagueError("EA IRIS conformance manifest identity or class is invalid")
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or manifest.get("fixtures_sha256") != _canonical_json_sha256(fixtures):
        raise ExternalLeagueError("EA IRIS conformance fixture ledger hash mismatches")
    observed_roles = {
        row.get("coverage_role") for row in fixtures if isinstance(row, Mapping)
    }
    observed_boundaries = {
        (boundary.get("family"), boundary.get("value"))
        for row in fixtures if isinstance(row, Mapping)
        for boundary in (row.get("temporal_boundary"),)
        if isinstance(boundary, Mapping)
    }
    expected_boundaries = {
        (family, value)
        for family, values in EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES.items()
        for value in values
    }
    if observed_roles != set(EA_IRIS_REQUIRED_CONFORMANCE_ROLES) or observed_boundaries != expected_boundaries:
        raise ExternalLeagueError("EA IRIS conformance mandatory role or temporal boundary coverage drifted")
    source_build_ref = manifest.get("source_build")
    source_video_ref = manifest.get("source_video_oracle")
    release_oracle_ref = manifest.get("release_oracle")
    if not all(isinstance(item, Mapping) for item in (source_build_ref, source_video_ref, release_oracle_ref)):
        raise ExternalLeagueError("EA IRIS conformance manifest build or release authority is invalid")
    source_build_path = Path(str(source_build_ref.get("path", ""))).resolve()
    if (
        not source_build_path.is_file()
        or source_build_ref.get("sha256") != _sha256_file(source_build_path)
    ):
        raise ExternalLeagueError("EA IRIS conformance source build receipt drifted")
    source_build, _ = _load_iris_source_build_receipt(source_build_path)
    if source_build_ref.get("binary_sha256") != source_build["binaries"]["source_frame_adapter"]["sha256"]:
        raise ExternalLeagueError("EA IRIS conformance source adapter binary differs from frozen build")
    if (
        source_video_ref.get("identity") != EA_IRIS_SOURCE_VIDEO_ORACLE_ID
        or source_video_ref.get("binary_sha256") != source_build["binaries"]["source_video_oracle"]["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS conformance source-video oracle differs from frozen build")
    release_census_path = Path(str(release_oracle_ref.get("census_receipt", ""))).resolve()
    if (
        release_oracle_ref.get("identity") != EA_IRIS_RELEASE_ORACLE_ID
        or not release_census_path.is_file()
        or release_oracle_ref.get("census_receipt_sha256") != _sha256_file(release_census_path)
        or not isinstance(release_oracle_ref.get("artifact_root"), str)
    ):
        raise ExternalLeagueError("EA IRIS conformance official release census authority drifted")
    release_census_entry, _ = _load_execution_census_entry(
        release_census_path,
        release_oracle_ref["artifact_root"],
        EA_IRIS_RELEASE_ORACLE_ID,
    )
    if release_oracle_ref.get("provenance_sha256") != release_census_entry["provenance_sha256"]:
        raise ExternalLeagueError("EA IRIS conformance official release provenance differs from census")
    expected_ids = [str(row["fixture_id"]) for row in fixtures]
    if (
        not isinstance(run_sets, Sequence)
        or isinstance(run_sets, (str, bytes))
        or len(run_sets) != len(fixtures)
    ):
        raise ExternalLeagueError("EA IRIS conformance run set coverage is incomplete")
    by_id: dict[str, Mapping[str, object]] = {}
    all_receipts: set[Path] = set()
    for row in run_sets:
        if not isinstance(row, Mapping) or set(row) != {
            "fixture_id", "source_adapter_runs", "source_video_runs", "release_oracle_runs",
        }:
            raise ExternalLeagueError("EA IRIS conformance run set fields are invalid")
        fixture_id = row.get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id in by_id:
            raise ExternalLeagueError("EA IRIS conformance run set identity is duplicate or invalid")
        role_refs = [row[field] for field in ("source_adapter_runs", "source_video_runs", "release_oracle_runs")]
        if any(
            not isinstance(refs, Sequence)
            or isinstance(refs, (str, bytes))
            or len(refs) != 3
            for refs in role_refs
        ):
            raise ExternalLeagueError("EA IRIS conformance requires exactly three ordered raw runs per role")
        refs = [Path(str(ref)).resolve() for role in role_refs for ref in role]
        if any(path in all_receipts for path in refs) or len(set(refs)) != 9:
            raise ExternalLeagueError("EA IRIS conformance cannot reuse one run receipt across roles or fixtures")
        all_receipts.update(refs)
        by_id[fixture_id] = row
    if set(by_id) != set(expected_ids):
        raise ExternalLeagueError("EA IRIS conformance run set does not match frozen fixture IDs")
    differential_rows: list[dict[str, object]] = []
    all_match = True
    for fixture in fixtures:
        fixture_id = str(fixture["fixture_id"])
        video = Path(str(fixture["canonical_video"]["path"])).resolve()
        conversion = Path(str(fixture["conversion_receipt"]["path"])).resolve()
        _, contract = _canonical_decoder_timeline_contract(video, conversion)
        if (
            fixture.get("fixture_class") != "CONFORMANCE_ONLY"
            or fixture.get("scoring_eligible") is not False
            or fixture["canonical_video"].get("sha256") != _sha256_file(video)
            or fixture["conversion_receipt"].get("sha256") != _sha256_file(conversion)
            or fixture.get("frame_map_sha256") != contract["frame_map_sha256"]
        ):
            raise ExternalLeagueError("EA IRIS frozen conformance fixture drifted")
        run_set = by_id[fixture_id]
        adapter_loaded = [
            _load_iris_source_adapter_run(ref, expected_lane="CONFORMANCE_ONLY")
            for ref in run_set["source_adapter_runs"]
        ]
        source_loaded = [_load_iris_source_video_run(ref) for ref in run_set["source_video_runs"]]
        release_loaded = [
            _load_iris_release_conformance_run(
                ref,
                video=video,
                conversion=conversion,
                census_entry=release_census_entry,
                census_path=release_census_path,
            )
            for ref in run_set["release_oracle_runs"]
        ]
        for repeat_ordinal, loaded_roles in enumerate(zip(adapter_loaded, source_loaded, release_loaded), start=1):
            adapter_receipt = loaded_roles[0][0]
            source_receipt = loaded_roles[1][0]
            release_receipt = loaded_roles[2][0]
            if (
                adapter_receipt["input"]["sha256"] != fixture["canonical_video"]["sha256"]
                or source_receipt["input"]["sha256"] != fixture["canonical_video"]["sha256"]
                or release_receipt["input"]["canonical_video_sha256"] != fixture["canonical_video"]["sha256"]
                or adapter_receipt["build_receipt"] != {"path": str(source_build_path), "sha256": _sha256_file(source_build_path)}
                or source_receipt["build_receipt"] != {"path": str(source_build_path), "sha256": _sha256_file(source_build_path)}
                or adapter_receipt["census_receipt"] != {"path": str(release_census_path), "sha256": _sha256_file(release_census_path)}
                or release_receipt.get("census_receipt") != {"path": str(release_census_path), "sha256": _sha256_file(release_census_path)}
                or adapter_receipt.get("fair_runtime", {}).get("scheduled_repeat_ordinal") != repeat_ordinal
                or source_receipt.get("fair_runtime", {}).get("scheduled_repeat_ordinal") != repeat_ordinal
                or release_receipt.get("fair_runtime", {}).get("scheduled_repeat_ordinal") != repeat_ordinal
            ):
                raise ExternalLeagueError("EA IRIS conformance run differs from frozen fixture, build, census, or repeat order")
        adapter_categories_by_repeat = [
            _iris_category_projection([
                {
                    "frame_index": row["frame_index"],
                    "luminance_result": row["native_frame_data"]["luminance_result"]["code"],
                    "red_result": row["native_frame_data"]["red_result"]["code"],
                    "pattern_result": row["native_frame_data"]["pattern_result"]["code"],
                }
                for row in raw["frames"]
            ])
            for _, raw, _, _ in adapter_loaded
        ]
        source_categories_by_repeat = [
            _iris_category_projection(detailed["categories"])
            for _, detailed, _ in source_loaded
        ]
        release_categories_by_repeat = [
            _iris_category_projection(detailed["categories"])
            for _, detailed, _ in release_loaded
        ]
        adapter_terminal_by_repeat = [observation for _, _, observation, _ in adapter_loaded]
        source_terminal_by_repeat = [
            {
                "prediction": detailed["prediction"],
                "warning": detailed["warning"],
                "hazard_frame_indices": detailed["hazard_frame_indices"],
                "warning_frame_indices": detailed["warning_frame_indices"],
            }
            for _, detailed, _ in source_loaded
        ]
        release_terminal_by_repeat = [
            {
                "prediction": detailed["prediction"],
                "warning": detailed["warning"],
                "hazard_frame_indices": detailed["hazard_frame_indices"],
                "warning_frame_indices": detailed["warning_frame_indices"],
            }
            for _, detailed, _ in release_loaded
        ]
        repeats_exact = all(
            len({_canonical_json_sha256(value) for value in values}) == 1
            for values in (
                adapter_categories_by_repeat,
                source_categories_by_repeat,
                release_categories_by_repeat,
                adapter_terminal_by_repeat,
                source_terminal_by_repeat,
                release_terminal_by_repeat,
            )
        )
        adapter_categories = adapter_categories_by_repeat[0]
        source_categories = source_categories_by_repeat[0]
        release_categories = release_categories_by_repeat[0]
        adapter_observation = adapter_terminal_by_repeat[0]
        source_detailed = source_loaded[0][1]
        release_detailed = release_loaded[0][1]
        coverage_role = fixture["coverage_role"]
        role_evidence = {
            "SAFE_CONTROL": all(
                row["luminance_result"] == 0
                and row["red_result"] == 0
                and row["pattern_result"] == 0
                for row in adapter_categories
            ),
            "LUMINANCE_FLASH": any(row["luminance_result"] >= 2 for row in adapter_categories),
            "RED_FLASH": any(row["red_result"] >= 2 for row in adapter_categories),
            "PATTERN": any(row["pattern_result"] >= 1 for row in adapter_categories),
        }[coverage_role]
        source_match = adapter_categories == source_categories
        release_match = adapter_categories == release_categories
        terminal_source_match = (
            adapter_observation["prediction"] == source_detailed["prediction"]
            and adapter_observation["warning"] == source_detailed["warning"]
            and adapter_observation["hazard_frame_indices"] == source_detailed["hazard_frame_indices"]
        )
        terminal_release_match = (
            adapter_observation["prediction"] == release_detailed["prediction"]
            and adapter_observation["warning"] == release_detailed["warning"]
            and adapter_observation["hazard_frame_indices"] == release_detailed["hazard_frame_indices"]
        )
        fixture_match = (
            role_evidence
            and repeats_exact
            and source_match
            and release_match
            and terminal_source_match
            and terminal_release_match
        )
        all_match = all_match and fixture_match
        differential_rows.append({
            "fixture_id": fixture_id,
            "coverage_role": fixture["coverage_role"],
            "temporal_boundary": fixture.get("temporal_boundary"),
            "coverage_role_verified_from_native_categories": role_evidence,
            "repeats_required": 3,
            "repeat_reproducibility_exact": repeats_exact,
            "source_adapter_runs": [
                {"path": str(path), "sha256": _sha256_file(path)} for _, _, _, path in adapter_loaded
            ],
            "source_video_runs": [
                {"path": str(path), "sha256": _sha256_file(path)} for _, _, path in source_loaded
            ],
            "release_oracle_runs": [
                {"path": str(path), "sha256": _sha256_file(path)} for _, _, path in release_loaded
            ],
            "adapter_categories_sha256": _canonical_json_sha256(adapter_categories),
            "source_video_categories_sha256": _canonical_json_sha256(source_categories),
            "release_categories_sha256": _canonical_json_sha256(release_categories),
            "adapter_vs_d969_source_video": {
                "frame_categories_exact": source_match,
                "terminal_exact": terminal_source_match,
            },
            "adapter_vs_official_1_1_0_release": {
                "frame_categories_exact": release_match,
                "terminal_exact": terminal_release_match,
            },
            "fixture_match": fixture_match,
        })
    receipt = {
        "schema": EA_IRIS_SOURCE_CONFORMANCE_RECEIPT_SCHEMA,
        "identity": EA_IRIS_SOURCE_ADAPTER_ID,
        "corpus_class": "CONFORMANCE_ONLY_NEVER_SCORING",
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "source_build": manifest["source_build"],
        "release_oracle": manifest["release_oracle"],
        "differential": differential_rows,
        "differential_sha256": _canonical_json_sha256(differential_rows),
        "local_fixture_match": all_match,
        "status": "LOCAL_CONFORMANCE_MATCH" if all_match else "NOT_VERIFIED",
        "release_equivalence": "NOT_CLAIMED",
        "official_release_role": "FROZEN_NON_SCORING_DIFFERENTIAL_ORACLE_ONLY",
        "execution_witness": "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT",
        "comparison_eligible": False,
        "scoreable": False,
        "scoreable_blockers": [
            *([] if all_match else ["source_or_release_conformance_drift"]),
            "local_execution_witness_not_independent",
            "independent_gold_receipt_missing",
            "frozen_public_case_ledger_missing",
        ],
    }
    _write_json(output, receipt)
    return {**receipt, "receipt": str(output)}


def _load_iris_source_conformance_receipt(
    receipt_ref: Path | str,
) -> tuple[dict[str, object], Path]:
    """Freshly replay every raw conformance reference before trusting the receipt."""
    receipt_path = Path(receipt_ref).resolve()
    try:
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS source conformance receipt is unreadable") from exc
    if (
        not isinstance(stored, Mapping)
        or stored.get("schema") != EA_IRIS_SOURCE_CONFORMANCE_RECEIPT_SCHEMA
        or stored.get("corpus_class") != "CONFORMANCE_ONLY_NEVER_SCORING"
        or stored.get("release_equivalence") != "NOT_CLAIMED"
        or stored.get("execution_witness") != "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT"
        or stored.get("comparison_eligible") is not False
        or stored.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS source conformance receipt class or claim boundary is invalid")
    manifest_ref = stored.get("manifest")
    differential = stored.get("differential")
    if (
        not isinstance(manifest_ref, Mapping)
        or not isinstance(manifest_ref.get("path"), str)
        or not isinstance(differential, list)
        or stored.get("differential_sha256") != _canonical_json_sha256(differential)
    ):
        raise ExternalLeagueError("EA IRIS source conformance receipt manifest or differential is invalid")
    manifest_path = Path(str(manifest_ref["path"])).resolve()
    if not manifest_path.is_file() or manifest_ref.get("sha256") != _sha256_file(manifest_path):
        raise ExternalLeagueError("EA IRIS source conformance frozen manifest drifted")
    run_sets: list[dict[str, object]] = []
    for row in differential:
        if not isinstance(row, Mapping):
            raise ExternalLeagueError("EA IRIS source conformance differential row is invalid")
        role_paths: dict[str, list[str]] = {}
        for field in ("source_adapter_runs", "source_video_runs", "release_oracle_runs"):
            refs = row.get(field)
            if not isinstance(refs, list) or len(refs) != 3:
                raise ExternalLeagueError("EA IRIS source conformance differential omits repeated raw receipts")
            paths: list[str] = []
            for reference in refs:
                if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
                    raise ExternalLeagueError("EA IRIS source conformance raw receipt reference is invalid")
                raw_path = Path(str(reference["path"])).resolve()
                if not raw_path.is_file() or reference.get("sha256") != _sha256_file(raw_path):
                    raise ExternalLeagueError("EA IRIS source conformance raw receipt hash drifted")
                paths.append(str(raw_path))
            role_paths[field] = paths
        run_sets.append({
            "fixture_id": row.get("fixture_id"),
            **role_paths,
        })
    with tempfile.TemporaryDirectory(prefix="flashpatch-iris-conformance-replay-") as temporary:
        replay_path = Path(temporary) / "receipt.json"
        replayed = verify_iris_source_adapter_conformance(manifest_path, run_sets, replay_path)
        replayed.pop("receipt", None)
    if dict(stored) != replayed:
        raise ExternalLeagueError("EA IRIS source conformance receipt differs from fresh raw replay")
    return dict(stored), receipt_path


def _load_conversion_receipt(path: Path, video: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("canonical conversion receipt is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "flashpatch-external-cfr-materialization-v1":
        raise ExternalLeagueError("canonical conversion receipt schema is invalid")
    canonical = payload.get("canonical_video")
    roundtrip = payload.get("roundtrip")
    if not isinstance(canonical, dict) or not isinstance(roundtrip, dict) or roundtrip.get("byte_identical") is not True:
        raise ExternalLeagueError("canonical conversion receipt lacks byte-identical roundtrip proof")
    if canonical.get("sha256") != _sha256_file(video):
        raise ExternalLeagueError("canonical video hash does not match conversion receipt")
    if (path.parent / str(canonical.get("path", ""))).resolve() != video:
        raise ExternalLeagueError("canonical video path is not owned by conversion receipt")
    return payload


def _checkout_provenance(spec: ComparatorSpec) -> dict[str, object]:
    if spec.source_checkout is None:
        return {"status": "UNVERIFIED", "reason": "source_checkout_not_declared"}
    checkout = spec.source_checkout.resolve()
    if not checkout.is_dir():
        return {"status": "UNVERIFIED", "reason": "source_checkout_unavailable", "path": str(checkout)}
    def run_git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, check=False)
    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    if head.returncode != 0 or status.returncode != 0:
        return {"status": "UNVERIFIED", "reason": "git_checkout_unreadable", "path": str(checkout)}
    revision = head.stdout.decode("utf-8", "replace").strip()
    clean = status.stdout == b""
    verified = revision == spec.revision and clean
    return {"status": "VERIFIED" if verified else "UNVERIFIED", "path": str(checkout), "head": revision, "clean": clean, "reason": None if verified else "revision_or_clean_tree_mismatch"}


def _require_cfr_timeline(timestamps: np.ndarray, fps: int) -> None:
    if fps <= 0:
        raise ExternalLeagueError("canonical CFR fps must be positive")
    expected = np.arange(len(timestamps), dtype=np.float64) / float(fps)
    # Renderer adapters may serialize either floor(index * 1e6 / fps) or a
    # floating seconds value.  Both describe the same CFR cadence; any larger
    # deviation is a VFR lane and must not be silently materialized for tools
    # that reconstruct time from average FPS.
    if not np.all(np.abs(timestamps - expected) <= 1.0e-6):
        raise ExternalLeagueError("external comparator lane requires exact CFR presentation timestamps")


def _decode_canonical_video_rgb(
    video: Path,
    conversion: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cache_key = (str(video.resolve()), str(conversion.resolve()))
    cached = _CANONICAL_DECODE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = _load_conversion_receipt(conversion, video)
    renderer = payload.get("renderer_rgb")
    cfr = payload.get("cfr")
    materializer = payload.get("materializer")
    if not all(isinstance(item, Mapping) for item in (renderer, cfr, materializer)):
        raise ExternalLeagueError("canonical conversion receipt omits decode provenance")
    shape = cfr.get("shape")
    fps = cfr.get("fps")
    frame_count = cfr.get("frame_count")
    ffmpeg = Path(str(materializer.get("binary", ""))).resolve()
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        or shape[-1] != 3
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or fps <= 0
        or frame_count != shape[0]
        or not ffmpeg.is_file()
        or materializer.get("binary_sha256") != _sha256_file(ffmpeg)
    ):
        raise ExternalLeagueError("canonical FFV1 decoder contract is invalid")
    command = [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    if materializer.get("decode_command") != command:
        raise ExternalLeagueError("canonical FFV1 fresh decode command differs from frozen conversion command")
    decoded = subprocess.run(command, capture_output=True, check=False)
    expected_bytes = int(np.prod(shape))
    if (
        decoded.returncode != 0
        or len(decoded.stdout) != expected_bytes
        or _sha256_bytes(decoded.stdout) != renderer.get("raw_sha256")
    ):
        raise ExternalLeagueError("canonical FFV1 decode does not reproduce renderer RGB")
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(tuple(shape)).copy()
    timestamps = np.arange(frame_count, dtype=np.float64) / float(fps)
    result = frames, timestamps, {
        "command": command,
        "exit_code": decoded.returncode,
        "decoder_sha256": _sha256_file(ffmpeg),
        "decoded_rgb_sha256": _sha256_bytes(decoded.stdout),
        "frame_count": frame_count,
        "fps": fps,
    }
    _CANONICAL_DECODE_CACHE[cache_key] = result
    return result


def _materialize_iris_direct_rgb_input(
    video: Path,
    conversion: Path,
    root: Path,
) -> dict[str, object]:
    """Freeze the exact canonical RGB slate consumed by the d969 API child.

    The FFmpeg command is the command already frozen by the lossless conversion
    receipt.  This materialization is intentionally outside the measured IRIS
    child, so it can prove G4 input identity but never G5 runtime parity.
    """
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    frames, _, decoder = _decode_canonical_video_rgb(video, conversion)
    if (
        contract["fps"] != 60
        or frames.shape != tuple(contract["shape"])
        or decoder["decoded_rgb_sha256"] != contract["renderer_source"]["rgb_sha256"]
        or decoder["frame_count"] != contract["frame_count"]
        or decoder["fps"] != contract["fps"]
        or decoder["exit_code"] != 0
    ):
        raise ExternalLeagueError("EA IRIS direct RGB materialization differs from canonical contract")
    raw_path = root / "canonical.rgb24"
    timeline_path = root / "canonical-rgb-timeline.json"
    raw_bytes = frames.tobytes()
    raw_path.write_bytes(raw_bytes)
    frame_rows = [
        {
            "frame_index": row["frame_index"],
            "cfr_timestamp": {"numerator": row["frame_index"], "denominator": 60},
            "cfr_timestamp_us": row["cfr_timestamp_us"],
            "renderer_timestamp_us": row["renderer_timestamp_us"],
            "rgb_sha256": row["rgb_sha256"],
        }
        for row in contract["frame_map"]
    ]
    if any(
        row["cfr_timestamp_us"] != row["renderer_timestamp_us"]
        for row in frame_rows
    ):
        raise ExternalLeagueError(
            "EA IRIS direct RGB adapter requires renderer timestamps exactly on the 60fps CFR grid"
        )
    conversion_payload = _load_conversion_receipt(conversion, video)
    materializer = conversion_payload.get("materializer")
    if not isinstance(materializer, Mapping):
        raise ExternalLeagueError("EA IRIS direct RGB input omits frozen FFmpeg materializer")
    ffmpeg = Path(str(materializer.get("binary", ""))).resolve()
    timeline = {
        "schema": "flashpatch-l7-ea-iris-direct-rgb-input-v1",
        "source_video": contract["canonical_video"],
        "conversion_receipt": contract["conversion_receipt"],
        "renderer_source": contract["renderer_source"],
        "decoder": {
            "binary": str(ffmpeg),
            "binary_sha256": decoder["decoder_sha256"],
            "command": decoder["command"],
            "decoded_rgb_sha256": decoder["decoded_rgb_sha256"],
            "frame_count": decoder["frame_count"],
            "fps": decoder["fps"],
            "exit_code": decoder["exit_code"],
        },
        "raw_rgb": {
            "path": str(raw_path),
            "sha256": _sha256_bytes(raw_bytes),
            "bytes": len(raw_bytes),
            "pixel_format": "rgb24",
            "shape": list(frames.shape),
        },
        "fps": contract["fps"],
        "frame_count": contract["frame_count"],
        "frames": frame_rows,
    }
    _write_json(timeline_path, timeline)
    raw_path.chmod(0o444)
    timeline_path.chmod(0o444)
    return {
        "raw_rgb": {
            "path": str(raw_path),
            "sha256": _sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        },
        "timeline": {
            "path": str(timeline_path),
            "sha256": _sha256_file(timeline_path),
        },
        "payload": timeline,
    }


def _load_iris_direct_rgb_input(
    raw_path: Path,
    timeline_path: Path,
    *,
    video: Path,
    conversion: Path,
    expected_raw_sha256: str | None = None,
    expected_timeline_sha256: str | None = None,
) -> dict[str, object]:
    """Freshly replay and verify the pre-materialized RGB input authority."""
    try:
        raw_bytes = raw_path.read_bytes()
        timeline_bytes = timeline_path.read_bytes()
        timeline = json.loads(timeline_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS direct RGB timeline is unreadable") from exc
    if (
        expected_raw_sha256 is not None
        and _sha256_bytes(raw_bytes) != expected_raw_sha256
    ):
        raise ExternalLeagueError("EA IRIS direct raw RGB bytes differ from receipt hash")
    if (
        expected_timeline_sha256 is not None
        and _sha256_bytes(timeline_bytes) != expected_timeline_sha256
    ):
        raise ExternalLeagueError("EA IRIS direct RGB timeline bytes differ from receipt hash")
    if not isinstance(timeline, Mapping) or set(timeline) != {
        "schema", "source_video", "conversion_receipt", "renderer_source",
        "decoder", "raw_rgb", "fps", "frame_count", "frames",
    }:
        raise ExternalLeagueError("EA IRIS direct RGB timeline fields are invalid")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    decoded, _, fresh_decoder = _decode_canonical_video_rgb(video, conversion)
    expected_rows = [
        {
            "frame_index": row["frame_index"],
            "cfr_timestamp": {"numerator": row["frame_index"], "denominator": 60},
            "cfr_timestamp_us": row["cfr_timestamp_us"],
            "renderer_timestamp_us": row["renderer_timestamp_us"],
            "rgb_sha256": row["rgb_sha256"],
        }
        for row in contract["frame_map"]
    ]
    raw_ref = timeline.get("raw_rgb")
    decoder = timeline.get("decoder")
    if not isinstance(raw_ref, Mapping) or set(raw_ref) != {
        "path", "sha256", "bytes", "pixel_format", "shape",
    }:
        raise ExternalLeagueError("EA IRIS direct raw RGB reference is invalid")
    if not isinstance(decoder, Mapping) or set(decoder) != {
        "binary", "binary_sha256", "command", "decoded_rgb_sha256",
        "frame_count", "fps", "exit_code",
    }:
        raise ExternalLeagueError("EA IRIS direct RGB decoder reference is invalid")
    if (
        timeline.get("schema") != "flashpatch-l7-ea-iris-direct-rgb-input-v1"
        or contract["fps"] != 60
        or timeline.get("source_video") != contract["canonical_video"]
        or timeline.get("conversion_receipt") != contract["conversion_receipt"]
        or timeline.get("renderer_source") != contract["renderer_source"]
        or timeline.get("fps") != contract["fps"]
        or timeline.get("frame_count") != contract["frame_count"]
        or timeline.get("frames") != expected_rows
        or any(row["cfr_timestamp_us"] != row["renderer_timestamp_us"] for row in expected_rows)
        or Path(str(raw_ref.get("path", ""))).resolve() != raw_path
        or not raw_path.is_file()
        or raw_ref.get("sha256") != _sha256_bytes(raw_bytes)
        or raw_ref.get("bytes") != len(raw_bytes)
        or raw_ref.get("pixel_format") != "rgb24"
        or raw_ref.get("shape") != contract["shape"]
        or raw_bytes != decoded.tobytes()
        or decoder != {
            "binary": fresh_decoder["command"][0],
            "binary_sha256": fresh_decoder["decoder_sha256"],
            "command": fresh_decoder["command"],
            "decoded_rgb_sha256": fresh_decoder["decoded_rgb_sha256"],
            "frame_count": fresh_decoder["frame_count"],
            "fps": fresh_decoder["fps"],
            "exit_code": fresh_decoder["exit_code"],
        }
    ):
        raise ExternalLeagueError("EA IRIS direct RGB input differs from fresh canonical replay")
    return dict(timeline)


def _flashpatch_fair_worker(video_arg: str, conversion_arg: str, output_arg: str) -> int:
    """Fresh-process worker used only by the fair L7 FlashPatch runtime lane."""
    video = Path(video_arg).resolve()
    conversion = Path(conversion_arg).resolve()
    output = Path(output_arg).resolve()
    try:
        frames, timestamps, decode_receipt = _decode_canonical_video_rgb(video, conversion)
        result = analyze(frames, timestamps)
        mask_path = output / "hazard-mask.npy"
        np.save(mask_path, result.hazard_mask)
        hazard_indices = np.flatnonzero(np.any(result.hazard_mask, axis=(1, 2))).astype(int).tolist()
        observation = {
            "tool": "FlashPatch",
            "prediction": "HAZARDOUS" if result.hazardous else "SAFE",
            "frame_count": len(frames),
            "fps": decode_receipt["fps"],
            "hazard_frame_indices": hazard_indices,
            "max_flash_count": result.max_flash_count,
            "max_affected_fraction": result.max_affected_fraction,
            "hazard_kinds": sorted(result.kind_masks),
            "timestamp_metrics": "canonical_cfr_timestamp",
            "mask_metrics": "hazard_mask_npy",
        }
        _write_json(output / "worker-observation.json", observation)
        _write_json(output / "worker-decode.json", decode_receipt)
        return 0
    except (ExternalLeagueError, OSError, ValueError) as exc:
        (output / "worker-error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return 2


def parse_tooflashy_json(raw_output: Path | str, canonical_video: Path | str, *, expected_fps: int, expected_frame_count: int) -> dict[str, object]:
    """Normalize TooFlashy's documented JSON result without inventing timing data.

    TooFlashy exposes only a case-level conclusion.  Its result therefore has
    no onset/mask score and can only enter the case-level detection lane.
    """
    raw = Path(raw_output)
    video = Path(canonical_video).resolve()
    try:
        payload = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("TooFlashy raw output is not JSON") from exc
    if not isinstance(payload, dict) or Path(str(payload.get("path", ""))).resolve() != video:
        raise ExternalLeagueError("TooFlashy result path does not match receipt-bound canonical video")
    passes, fps, frame_count, event_count, failures = (payload.get(name) for name in ("passes", "fps", "frame_count", "event_count", "failures"))
    if not isinstance(passes, bool) or isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != float(expected_fps):
        raise ExternalLeagueError("TooFlashy result FPS does not match canonical CFR lane")
    if not isinstance(frame_count, int) or frame_count != expected_frame_count or not isinstance(event_count, int) or event_count < 0 or not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise ExternalLeagueError("TooFlashy result does not provide a valid case-level observation")
    if passes is False and not failures:
        raise ExternalLeagueError("TooFlashy failing result must name at least one failure")
    return {"tool": "TooFlashy", "prediction": "SAFE" if passes else "HAZARDOUS", "frame_count": frame_count, "fps": float(fps), "event_count": event_count, "failures": failures, "timestamp_metrics": "NOT_APPLICABLE", "mask_metrics": "NOT_APPLICABLE"}


def _verify_tooflashy_repeat_checkout(checkout: Path) -> None:
    """Require the exact clean TooFlashy source revision used by a staged run."""
    if not checkout.is_dir():
        raise ExternalLeagueError("TooFlashy repeat checkout is missing")
    commands = {
        "revision": ["/usr/bin/git", "rev-parse", "HEAD"],
        "tree": ["/usr/bin/git", "rev-parse", "HEAD^{tree}"],
        "status": ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"],
    }
    observations: dict[str, str] = {}
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                cwd=checkout,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalLeagueError(f"TooFlashy repeat checkout audit failed: {name}") from exc
        if completed.returncode != 0:
            raise ExternalLeagueError(f"TooFlashy repeat checkout audit failed: {name}")
        observations[name] = completed.stdout.strip()
    if (
        observations["revision"] != TOOFLASHY_PARITY_ADAPTER_REVISION
        or observations["tree"] != TOOFLASHY_PARITY_ADAPTER_TREE
    ):
        raise ExternalLeagueError("TooFlashy repeat checkout revision is not pinned")
    if observations["status"]:
        raise ExternalLeagueError("TooFlashy repeat checkout is not clean")


def verify_tooflashy_oldfilm_repeats(
    canonical_source: Path | str,
    run_roots: Sequence[Path | str],
) -> str:
    """Reparse exactly three staged OldFilm runs and keep them non-scoreable."""
    source = Path(canonical_source).resolve()
    if not source.is_file():
        raise ExternalLeagueError("TooFlashy OldFilm canonical source is missing")
    source_sha256 = _sha256_file(source)
    if source_sha256 != TOOFLASHY_OLDFILM_CANONICAL_SHA256:
        raise ExternalLeagueError("TooFlashy OldFilm canonical source hash mismatch")
    if (
        not isinstance(run_roots, Sequence)
        or isinstance(run_roots, (str, bytes))
        or len(run_roots) != 3
    ):
        raise ExternalLeagueError("TooFlashy OldFilm requires exactly three repeats")
    roots = [Path(root).resolve() for root in run_roots]
    if len(set(roots)) != 3:
        raise ExternalLeagueError("TooFlashy OldFilm repeat roots must be distinct")

    observations: list[dict[str, object]] = []
    for root in roots:
        staged_video = root / "canonical.ffv1.mkv"
        raw_output = root / "tooflashy.json"
        if not staged_video.is_file() or _sha256_file(staged_video) != source_sha256:
            raise ExternalLeagueError("TooFlashy OldFilm staged input hash mismatch")
        _verify_tooflashy_repeat_checkout(root / "repo")
        try:
            raw_payload = json.loads(raw_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError("TooFlashy OldFilm raw output is unreadable") from exc
        if not isinstance(raw_payload, Mapping) or set(raw_payload) != TOOFLASHY_OFFICIAL_JSON_FIELDS:
            raise ExternalLeagueError("TooFlashy OldFilm raw output fields are invalid")
        observations.append(
            parse_tooflashy_json(
                raw_output,
                staged_video,
                expected_fps=60,
                expected_frame_count=150,
            )
        )
    if len({_canonical_json_sha256(observation) for observation in observations}) != 1:
        raise ExternalLeagueError("TooFlashy OldFilm repeat observations disagree")
    return "NOT_SCOREABLE"


def parse_tooflashy_adapter_json(
    raw_output: Path | str,
    canonical_video: Path | str,
    *,
    expected_fps: int,
    expected_frame_count: int,
) -> dict[str, object]:
    """Normalize only a same-process, source-hash-bound adapter payload."""
    path = Path(raw_output).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("TooFlashy adapter raw output is not JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {
            "schema", "evidence_origin", "adapter_source_sha256", "input",
            "public_api", "runtime", "decode", "result",
        }
        or payload.get("schema") != "flashpatch-l7-tooflashy-child-adapter-v1"
        or payload.get("evidence_origin") != "live_generator_append_immediately_before_yield_v1"
        or payload.get("adapter_source_sha256")
        != _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
    ):
        raise ExternalLeagueError("TooFlashy adapter raw evidence identity is invalid")
    result = payload.get("result")
    decode = payload.get("decode")
    input_ref = payload.get("input")
    video = Path(canonical_video).resolve()
    if (
        not isinstance(result, Mapping)
        or set(result) != {*TOOFLASHY_OFFICIAL_JSON_FIELDS, "event_representation"}
        or not isinstance(decode, Mapping)
        or not isinstance(input_ref, Mapping)
        or input_ref != {"path": str(video), "sha256": _sha256_file(video)}
        or result.get("path") != str(video)
    ):
        raise ExternalLeagueError("TooFlashy adapter result or input binding is invalid")
    passes = result.get("passes")
    fps = result.get("fps")
    frame_count = result.get("frame_count")
    event_count = result.get("event_count")
    failures = result.get("failures")
    if (
        not isinstance(passes, bool)
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) != float(expected_fps)
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != expected_frame_count
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 0
        or not isinstance(failures, list)
        or not all(isinstance(item, str) for item in failures)
        or result.get("event_representation")
        != {"event_count": event_count, "failures": failures}
        or decode.get("fps") != fps
        or decode.get("frame_count") != frame_count
    ):
        raise ExternalLeagueError("TooFlashy adapter outcome differs from canonical CFR contract")
    return {
        "tool": "TooFlashy",
        "prediction": "SAFE" if passes else "HAZARDOUS",
        "frame_count": frame_count,
        "fps": float(fps),
        "event_count": event_count,
        "failures": failures,
        "timestamp_metrics": "canonical_cfr_preconsumption_ledger",
        "mask_metrics": "NOT_APPLICABLE",
    }


def parse_iris_json(result_path: Path | str, frame_data_path: Path | str, *, expected_frame_count: int) -> dict[str, object]:
    """Normalize legacy source JSON without minting the strict adapter identity."""
    try:
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        frame_data = json.loads(Path(frame_data_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("EA IRIS output JSON is unreadable") from exc
    if not isinstance(result, dict) or not isinstance(frame_data, dict):
        raise ExternalLeagueError("EA IRIS output JSON roots must be objects")
    overall = result.get("OverallResult")
    total = result.get("TotalFrame")
    if overall not in {"Pass", "PassWithWarning", "Fail"} or total != expected_frame_count:
        raise ExternalLeagueError("EA IRIS case result does not match the canonical frame count")
    graph = frame_data.get("LineGraphFrameData")
    if not isinstance(graph, dict):
        raise ExternalLeagueError("EA IRIS frameData omits LineGraphFrameData")
    fields = ("LuminanceFrameResult", "RedFrameResult", "PatternFrameResult")
    arrays = [graph.get(field) for field in fields]
    if any(not isinstance(values, list) or len(values) != expected_frame_count or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values) for values in arrays):
        raise ExternalLeagueError("EA IRIS frame result arrays do not match the canonical frame count")
    luminance, red, pattern = arrays
    hazardous_indices = [index for index in range(expected_frame_count) if luminance[index] >= 2 or red[index] >= 2 or pattern[index] >= 1]
    return {
        "tool": EA_IRIS_LEGACY_JSON_ID,
        "prediction": "HAZARDOUS" if overall == "Fail" else "SAFE",
        "warning": overall == "PassWithWarning",
        "frame_count": total,
        "hazard_frame_indices": hazardous_indices,
        "timestamp_metrics": "frame_index_only",
        "mask_metrics": "NOT_APPLICABLE",
        "comparison_eligible": False,
        "comparison_blocker": "source_build_frame_ledger_and_release_conformance_missing",
    }


def parse_iris_release_csv(
    csv_path: Path | str,
    stdout_path: Path | str,
    *,
    expected_frame_count: int,
    expected_fps: int,
) -> dict[str, object]:
    """Normalize the official EA IRIS 1.1.0 Ubuntu example-app report.

    The release executable writes a NUL-terminated CSV rather than the JSON
    files produced by the source example app.  The parser binds the terminal
    result in its raw stdout to a sequential per-frame report and to the
    comparator-observed FPS/frame count.  It does not convert IRIS's rounded
    CSV timestamps into a false claim of sub-frame timing precision.
    """
    try:
        raw_csv = Path(csv_path).read_bytes().replace(b"\x00", b"")
        stdout = Path(stdout_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ExternalLeagueError("EA IRIS release output is unreadable") from exc
    try:
        rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ExternalLeagueError("EA IRIS release frame report is invalid CSV") from exc
    required = {"Frame", "TimeStamp", "LuminanceFrameResult", "RedFrameResult", "PatternFrameResult"}
    if not rows or set(rows[0]) < required or len(rows) != expected_frame_count:
        raise ExternalLeagueError("EA IRIS release frame report does not match the canonical frame count")
    try:
        frames = [int(row["Frame"]) for row in rows]
        luminance = [int(row["LuminanceFrameResult"]) for row in rows]
        red = [int(row["RedFrameResult"]) for row in rows]
        pattern = [int(row["PatternFrameResult"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalLeagueError("EA IRIS release frame results are invalid") from exc
    if frames != list(range(1, expected_frame_count + 1)) or any(value < 0 for value in luminance + red + pattern):
        raise ExternalLeagueError("EA IRIS release frame numbering or results are invalid")
    fps_match = re.search(r"^.*Video FPS:\s*(\d+)\s*$", stdout, flags=re.MULTILINE)
    frame_match = re.search(r"^.*Total frames:\s*(\d+)\s*$", stdout, flags=re.MULTILINE)
    result_match = re.search(r"^.*Video Result:\s*(PASS|FAIL)\s*$", stdout, flags=re.MULTILINE)
    if fps_match is None or int(fps_match.group(1)) != expected_fps or frame_match is None or int(frame_match.group(1)) != expected_frame_count or result_match is None:
        raise ExternalLeagueError("EA IRIS release stdout does not match the canonical CFR input")
    hazardous_indices = [index for index, values in enumerate(zip(luminance, red, pattern)) if values[0] >= 2 or values[1] >= 2 or values[2] >= 1]
    if (result_match.group(1) == "FAIL") != bool(hazardous_indices):
        raise ExternalLeagueError("EA IRIS release terminal result conflicts with its frame report")
    return {
        "tool": EA_IRIS_RELEASE_ORACLE_ID,
        "distribution": "official-ubuntu-example-app-1.1.0",
        "prediction": "HAZARDOUS" if result_match.group(1) == "FAIL" else "SAFE",
        "frame_count": expected_frame_count,
        "fps": expected_fps,
        "hazard_frame_indices": hazardous_indices,
        "timestamp_metrics": "rounded_csv_timestamp_only",
        "mask_metrics": "NOT_APPLICABLE",
    }


def execute_iris_release(
    spec: IrisReleaseSpec,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    census_receipt: Path | str | None = None,
    census_artifact_root: Path | str | None = None,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
    scheduled_repeat_ordinal: int | None = None,
    runtime_schedule: Path | str | None = None,
    schedule_slot: int | None = None,
) -> dict[str, object]:
    """Run the official IRIS Ubuntu example-app against one receipt-owned lane.

    This is a non-scoring release-binary conformance-oracle path, distinct from a local source
    checkout.  It records the signed-release asset hash, staged executable and
    configuration hashes, raw stdout/stderr and per-frame CSV before parsing.
    The resulting observation proves process, decode and parser binding only;
    It can never stand in for the source-frame adapter in the direct detector
    population or contribute an L7 score.
    """
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS release output already exists: {root}")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    conversion_payload = _load_conversion_receipt(conversion, video)
    schedule_binding = _load_schedule_assignment(
        runtime_schedule,
        schedule_slot=schedule_slot,
        protocol=frozen_runtime,
        comparator=EA_IRIS_RELEASE_ORACLE_ID,
        repeat_ordinal=scheduled_repeat_ordinal,
        input_sha256=_sha256_file(video),
    )
    if not spec.repository_url or re.fullmatch(r"[0-9a-f]{40}", spec.source_revision) is None or not spec.release_tag:
        raise ExternalLeagueError("EA IRIS release provenance is incomplete")
    if not spec.release_asset.is_file() or _sha256_file(spec.release_asset) != spec.release_asset_sha256:
        raise ExternalLeagueError("EA IRIS release asset hash does not match the declared release")
    if not spec.executable.is_file() or not spec.appsettings.is_file() or not isinstance(spec.expected_fps, int) or spec.expected_fps <= 0:
        raise ExternalLeagueError("EA IRIS release executable contract is incomplete")
    if census_receipt is None or census_artifact_root is None:
        raise ExternalLeagueError("EA IRIS official release requires a validated census receipt")
    census_entry, census_path = _load_execution_census_entry(
        census_receipt,
        census_artifact_root,
        EA_IRIS_RELEASE_ORACLE_ID,
    )
    for field, actual in (
        ("repository_url", spec.repository_url),
        ("distribution_source_revision", spec.source_revision),
        ("distribution_revision", spec.release_tag),
        ("release_asset_sha256", spec.release_asset_sha256),
        ("binary_sha256", _sha256_file(spec.executable)),
        ("configuration_sha256", _sha256_file(spec.appsettings)),
    ):
        if census_entry[field] != actual:
            raise ExternalLeagueError(f"EA IRIS release execution differs from census provenance: {field}")
    frame_count = conversion_payload["cfr"].get("frame_count")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ExternalLeagueError("canonical conversion receipt frame count is invalid")
    root.mkdir(parents=True)
    staged_binary = root / "IrisApp"
    staged_settings = root / "appsettings.json"
    staged_video = root / "TestVideos" / "canonical.ffv1.mkv"
    results = root / "Results" / staged_video.name
    staged_video.parent.mkdir()
    results.mkdir(parents=True)
    shutil.copy2(spec.executable, staged_binary)
    shutil.copy2(spec.appsettings, staged_settings)
    staged_binary.chmod(staged_binary.stat().st_mode | 0o111)
    tool_command = [str(staged_binary), "-j", str(staged_video.relative_to(root)), "-p", "1", "-r", "0"]
    with _fair_execution_context(
        frozen_runtime,
        video,
        base_environment=os.environ,
        schedule_binding=schedule_binding,
    ) as execution:
        probe_path = root / "runtime-probe.json"
        command = _instrument_fair_command(execution, tool_command, probe_path)
        started = time.monotonic_ns()
        conversion_payload = _load_conversion_receipt(conversion, video)
        shutil.copy2(video, staged_video)
        timeout_seconds = int(frozen_runtime["budget"]["timeout_seconds"]) if frozen_runtime is not None else 120
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env=execution["environment"],
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: EA IRIS timeout",
            )
        stdout_path = root / "stdout.bin"
        stderr_path = root / "stderr.bin"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        csv_path = results / "framedata.csv"
        parsed: dict[str, object] | None = None
        parse_error: str | None = None
        if completed.returncode == 0 and csv_path.is_file():
            try:
                parsed = parse_iris_release_csv(
                    csv_path,
                    stdout_path,
                    expected_frame_count=frame_count,
                    expected_fps=spec.expected_fps,
                )
            except ExternalLeagueError as exc:
                parse_error = str(exc)
        child_probe: dict[str, object] | None = None
        if frozen_runtime is not None:
            try:
                child_probe = _load_child_runtime_probe(probe_path)
            except ExternalLeagueError:
                pass
        finished = time.monotonic_ns()
        elapsed_ns = finished - started
        if elapsed_ns > timeout_seconds * 1_000_000_000:
            timed_out = True
        status = "PROCESS_VALID" if parsed is not None and not timed_out and (frozen_runtime is None or child_probe is not None) else "INCONCLUSIVE"
        runtime_receipt = _fair_runtime_run_receipt(
            frozen_runtime,
            comparator=EA_IRIS_RELEASE_ORACLE_ID,
            scheduled_repeat_ordinal=scheduled_repeat_ordinal,
            schedule_binding=schedule_binding,
            input_sha256=_sha256_file(video),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            wall_time_ns=elapsed_ns,
            timed_out=timed_out,
            observation=parsed,
            normalizer="ea-iris-release-csv-v1",
            observed_environment={
                "parent_precondition": execution["observation"],
                "child_probe": child_probe,
            } if frozen_runtime is not None else None,
        )
    receipt = {
        "schema": "flashpatch-ea-iris-release-run-v1",
        "comparator": {
            "name": EA_IRIS_RELEASE_ORACLE_ID,
            "repository_url": spec.repository_url,
            "source_revision": spec.source_revision,
            "release_tag": spec.release_tag,
            "release_asset": {"path": str(spec.release_asset.resolve()), "sha256": spec.release_asset_sha256},
            "executable_sha256": _sha256_file(staged_binary),
            "appsettings_sha256": _sha256_file(staged_settings),
        },
        "input": {"sha256": _sha256_file(staged_video), "canonical_video_sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion), "renderer_rgb_sha256": conversion_payload["renderer_rgb"]["raw_sha256"]},
        "census_receipt": {"path": str(census_path), "sha256": _sha256_file(census_path)},
        "command": command,
        "exit_code": completed.returncode,
        "wall_time_ns": elapsed_ns,
        "fair_runtime": runtime_receipt,
        "runtime_probe": {
            "path": probe_path.name,
            "sha256": _sha256_file(probe_path) if frozen_runtime is not None and probe_path.is_file() else None,
            "observation": child_probe if frozen_runtime is not None else None,
        },
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
        "frame_report": {"path": str(csv_path.relative_to(root)), "exists": csv_path.is_file(), "sha256": _sha256_file(csv_path) if csv_path.is_file() else None},
        "parsed_observation": parsed,
        "parse_error": parse_error,
        "status": status,
        "scoreable": False,
        "scoreable_blockers": ["independent_gold_receipt_missing", "frozen_public_case_ledger_missing"] if status == "PROCESS_VALID" else ["release_execution_or_parser_inconclusive"],
    }
    receipt_path = root / "iris-release-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def execute_repeated_iris_release(
    spec: IrisReleaseSpec,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    repeats: int = 3,
    census_receipt: Path | str | None = None,
    census_artifact_root: Path | str | None = None,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require three independent official-release runs before reproducibility.

    Raw console logs contain wall-clock timestamps, so this function requires
    equality of the per-frame report hash and canonical parsed observation
    instead of pretending those volatile logs are byte-identical.
    """
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"EA IRIS release repeat output already exists: {root}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 3:
        raise ExternalLeagueError("L7 requires exactly three EA IRIS release repeats")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen_runtime is not None and frozen_runtime["budget"]["scheduled_repeats"] != repeats:
        raise ExternalLeagueError("EA IRIS repeat count differs from fair runtime budget")
    root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    for index in range(1, repeats + 1):
        try:
            result = execute_iris_release(
                spec,
                canonical_video,
                conversion_receipt,
                root / f"run-{index}",
                census_receipt=census_receipt,
                census_artifact_root=census_artifact_root,
                runtime_protocol=frozen_runtime,
                scheduled_repeat_ordinal=index if frozen_runtime is not None else None,
            )
            observation = result.get("parsed_observation")
            runs.append({
                "repeat": index,
                "status": result["status"],
                "receipt": result["receipt"],
                "receipt_sha256": _sha256_file(Path(str(result["receipt"]))),
                "frame_report_sha256": result["frame_report"]["sha256"],
                "fair_runtime": result.get("fair_runtime"),
                "observation_sha256": _sha256_bytes(json.dumps(observation, sort_keys=True, separators=(",", ":")).encode("utf-8")) if isinstance(observation, dict) else None,
            })
        except (ExternalLeagueError, OSError, subprocess.SubprocessError) as exc:
            runs.append({"repeat": index, "status": "INCONCLUSIVE", "reason": str(exc)})
    frame_hashes = {run.get("frame_report_sha256") for run in runs if run.get("status") == "PROCESS_VALID"}
    observations = {run.get("observation_sha256") for run in runs if run.get("status") == "PROCESS_VALID"}
    reproducible = len(runs) == repeats and all(run.get("status") == "PROCESS_VALID" for run in runs) and len(frame_hashes) == 1 and None not in frame_hashes and len(observations) == 1 and None not in observations
    receipt = {
        "schema": "flashpatch-ea-iris-release-conformance-repeats-v1",
        "repeats_required": 3,
        "comparator": EA_IRIS_RELEASE_ORACLE_ID,
        "fair_runtime_protocol": frozen_runtime,
        "fair_runtime_protocol_sha256": _canonical_json_sha256(frozen_runtime) if frozen_runtime is not None else None,
        "runs": runs,
        "status": "PROCESS_REPRODUCIBLE" if reproducible else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": ["independent_gold_receipt_missing", "frozen_public_case_ledger_missing"],
    }
    receipt_path = root / "iris-release-repeat-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def execute_flashpatch_detector(
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
    scheduled_repeat_ordinal: int | None = None,
    runtime_schedule: Path | str | None = None,
    schedule_slot: int | None = None,
) -> dict[str, object]:
    """Record FlashPatch's direct RGB observation for an external L7 lane.

    FlashPatch consumes the renderer-owned NPZ that was independently
    round-tripped into the canonical FFV1 video.  The conversion receipt binds
    those representations through one raw RGB hash, so this direct input path
    does not silently give FlashPatch a different frame slate.
    """
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"FlashPatch detector output already exists: {root}")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    schedule_binding = _load_schedule_assignment(
        runtime_schedule,
        schedule_slot=schedule_slot,
        protocol=frozen_runtime,
        comparator="FlashPatch",
        repeat_ordinal=scheduled_repeat_ordinal,
        input_sha256=_sha256_file(video),
    )
    if frozen_runtime is not None:
        conversion_payload = _load_conversion_receipt(conversion, video)
        renderer = conversion_payload.get("renderer_rgb")
        if not isinstance(renderer, Mapping):
            raise ExternalLeagueError("canonical conversion receipt omits renderer RGB identity")
        root.mkdir(parents=True)
        with _fair_execution_context(
            frozen_runtime,
            video,
            base_environment=os.environ,
            schedule_binding=schedule_binding,
            launcher_cwd=Path(__file__).resolve().parents[2],
        ) as execution:
            worker_command = [
                sys.executable,
                "-c",
                _FLASHPATCH_WORKER_SCRIPT,
                str(video),
                str(conversion),
                str(root),
            ]
            probe_path = root / "runtime-probe.json"
            command = _instrument_fair_command(execution, worker_command, probe_path)
            started = time.monotonic_ns()
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    timeout=int(frozen_runtime["budget"]["timeout_seconds"]),
                    env=execution["environment"],
                    cwd=str(Path(__file__).resolve().parents[2]),
                )
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                completed = subprocess.CompletedProcess(
                    command,
                    124,
                    exc.stdout or b"",
                    (exc.stderr or b"") + b"\nflashpatch: detector timeout",
                )
            stdout_path = root / "stdout.bin"
            stderr_path = root / "stderr.bin"
            stdout_path.write_bytes(completed.stdout)
            stderr_path.write_bytes(completed.stderr)
            observation_path = root / "worker-observation.json"
            decode_path = root / "worker-decode.json"
            mask_path = root / "hazard-mask.npy"
            observation: dict[str, object] | None = None
            decode_receipt: dict[str, object] | None = None
            child_probe: dict[str, object] | None = None
            try:
                loaded_observation = json.loads(observation_path.read_text(encoding="utf-8"))
                loaded_decode = json.loads(decode_path.read_text(encoding="utf-8"))
                if isinstance(loaded_observation, dict) and isinstance(loaded_decode, dict):
                    observation = loaded_observation
                    decode_receipt = loaded_decode
            except (OSError, json.JSONDecodeError):
                pass
            try:
                child_probe = _load_child_runtime_probe(probe_path)
            except ExternalLeagueError:
                pass
            finished = time.monotonic_ns()
            elapsed_ns = finished - started
            if elapsed_ns > int(frozen_runtime["budget"]["timeout_seconds"]) * 1_000_000_000:
                timed_out = True
            process_valid = (
                completed.returncode == 0
                and not timed_out
                and observation is not None
                and decode_receipt is not None
                and mask_path.is_file()
                and child_probe is not None
            )
            runtime_receipt = _fair_runtime_run_receipt(
                frozen_runtime,
                comparator="FlashPatch",
                scheduled_repeat_ordinal=scheduled_repeat_ordinal,
                schedule_binding=schedule_binding,
                input_sha256=_sha256_file(video),
                started_monotonic_ns=started,
                finished_monotonic_ns=finished,
                wall_time_ns=elapsed_ns,
                timed_out=timed_out,
                observation=observation,
                normalizer="flashpatch-direct-detector-v1",
                observed_environment={
                    "parent_precondition": execution["observation"],
                    "child_probe": child_probe,
                },
            )
        receipt = {
            "schema": "flashpatch-l7-direct-detector-run-v1",
            "comparator": {
                "name": "FlashPatch",
                "code_path": str(Path(flashpatch_core.__file__).resolve()),
                "code_sha256": _sha256_file(Path(flashpatch_core.__file__)),
            },
            "input": {
                "canonical_video_sha256": _sha256_file(video),
                "renderer_rgb_sha256": renderer.get("raw_sha256"),
            },
            "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
            "command": command,
            "exit_code": completed.returncode,
            "stdout_sha256": _sha256_file(stdout_path),
            "stderr_sha256": _sha256_file(stderr_path),
            "observation": observation,
            "worker_decode": {
                "path": decode_path.name,
                "sha256": _sha256_file(decode_path) if decode_path.is_file() else None,
                "receipt": decode_receipt,
            },
            "runtime_probe": {
                "path": probe_path.name,
                "sha256": _sha256_file(probe_path) if probe_path.is_file() else None,
                "observation": child_probe,
            },
            "hazard_mask": {
                "path": mask_path.name,
                "sha256": _sha256_file(mask_path) if mask_path.is_file() else None,
                "shape": list(np.load(mask_path).shape) if mask_path.is_file() else None,
            },
            "wall_time_ns": elapsed_ns,
            "fair_runtime": runtime_receipt,
            "status": "PROCESS_VALID" if process_valid else "INCONCLUSIVE",
            "scoreable": False,
            "scoreable_blockers": (
                ["runtime_execution_or_normalization_inconclusive"]
                if not process_valid
                else ["independent_gold_receipt_missing", "frozen_public_case_ledger_missing"]
            ),
        }
        receipt_path = root / "flashpatch-detector-receipt.json"
        _write_json(receipt_path, receipt)
        return {**receipt, "receipt": str(receipt_path)}

    started = time.monotonic_ns()
    payload = _load_conversion_receipt(conversion, video)
    source_info = payload.get("source")
    cfr = payload.get("cfr")
    renderer = payload.get("renderer_rgb")
    if not isinstance(source_info, dict) or not isinstance(cfr, dict) or not isinstance(renderer, dict):
        raise ExternalLeagueError("canonical conversion receipt is incomplete for FlashPatch detection")
    source = Path(str(source_info.get("path", ""))).resolve()
    if not source.is_file() or source_info.get("sha256") != _sha256_file(source):
        raise ExternalLeagueError("renderer NPZ source does not match canonical conversion receipt")
    try:
        with np.load(source) as archive:
            frames = np.asarray(archive["frames"])
            timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as exc:
        raise ExternalLeagueError("renderer NPZ source is unreadable") from exc
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3 or timestamps.shape != (len(frames),):
        raise ExternalLeagueError("renderer NPZ source does not contain valid RGB frames")
    if _sha256_bytes(frames.tobytes()) != renderer.get("raw_sha256") or len(frames) != cfr.get("frame_count"):
        raise ExternalLeagueError("renderer NPZ RGB bytes do not match canonical conversion receipt")
    root.mkdir(parents=True)
    result = analyze(frames, timestamps)
    mask_path = root / "hazard-mask.npy"
    np.save(mask_path, result.hazard_mask)
    hazard_indices = np.flatnonzero(np.any(result.hazard_mask, axis=(1, 2))).astype(int).tolist()
    observation = {
        "tool": "FlashPatch",
        "prediction": "HAZARDOUS" if result.hazardous else "SAFE",
        "frame_count": len(frames),
        "fps": cfr.get("fps"),
        "hazard_frame_indices": hazard_indices,
        "max_flash_count": result.max_flash_count,
        "max_affected_fraction": result.max_affected_fraction,
        "hazard_kinds": sorted(result.kind_masks),
        "timestamp_metrics": "renderer_monotonic_timestamp",
        "mask_metrics": "hazard_mask_npy",
    }
    finished = time.monotonic_ns()
    elapsed_ns = finished - started
    timed_out = bool(
        frozen_runtime is not None
        and elapsed_ns > int(frozen_runtime["budget"]["timeout_seconds"]) * 1_000_000_000
    )
    runtime_receipt = _fair_runtime_run_receipt(
        frozen_runtime,
        comparator="FlashPatch",
        scheduled_repeat_ordinal=scheduled_repeat_ordinal,
        schedule_binding=schedule_binding,
        input_sha256=_sha256_file(video),
        started_monotonic_ns=started,
        finished_monotonic_ns=finished,
        wall_time_ns=elapsed_ns,
        timed_out=timed_out,
        observation=observation,
        normalizer="flashpatch-direct-detector-v1",
        observed_environment=None,
    )
    receipt = {
        "schema": "flashpatch-l7-direct-detector-run-v1",
        "comparator": {"name": "FlashPatch", "code_path": str(Path(flashpatch_core.__file__).resolve()), "code_sha256": _sha256_file(Path(flashpatch_core.__file__))},
        "input": {"canonical_video_sha256": _sha256_file(video), "renderer_npz_sha256": _sha256_file(source), "renderer_rgb_sha256": renderer["raw_sha256"]},
        "conversion_receipt": {"sha256": _sha256_file(conversion)},
        "observation": observation,
        "hazard_mask": {"path": mask_path.name, "sha256": _sha256_file(mask_path), "shape": list(result.hazard_mask.shape)},
        "wall_time_ns": elapsed_ns,
        "fair_runtime": runtime_receipt,
        "status": "INCONCLUSIVE" if timed_out else "PROCESS_VALID",
        "scoreable": False,
        "scoreable_blockers": (
            ["runtime_timeout"]
            if timed_out
            else ["independent_gold_receipt_missing", "frozen_public_case_ledger_missing"]
        ),
    }
    receipt_path = root / "flashpatch-detector-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def execute_repeated_flashpatch_detector(
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    repeats: int = 3,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require three reproducible direct FlashPatch observations for one lane."""
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"FlashPatch detector repeat output already exists: {root}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 3:
        raise ExternalLeagueError("L7 requires exactly three FlashPatch detector repeats")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen_runtime is not None and frozen_runtime["budget"]["scheduled_repeats"] != repeats:
        raise ExternalLeagueError("FlashPatch repeat count differs from fair runtime budget")
    root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    for index in range(1, repeats + 1):
        try:
            run = execute_flashpatch_detector(
                canonical_video,
                conversion_receipt,
                root / f"run-{index}",
                runtime_protocol=frozen_runtime,
                scheduled_repeat_ordinal=index if frozen_runtime is not None else None,
            )
            observation = run.get("observation")
            runs.append({
                "repeat": index,
                "status": run["status"],
                "receipt": run["receipt"],
                "hazard_mask_sha256": run["hazard_mask"]["sha256"],
                "receipt_sha256": _sha256_file(Path(str(run["receipt"]))),
                "fair_runtime": run.get("fair_runtime"),
                "observation_sha256": _sha256_bytes(
                    json.dumps(observation, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ) if isinstance(observation, dict) else None,
            })
        except (ExternalLeagueError, OSError, ValueError) as exc:
            runs.append({"repeat": index, "status": "INCONCLUSIVE", "reason": str(exc)})
    mask_hashes = {run.get("hazard_mask_sha256") for run in runs if run.get("status") == "PROCESS_VALID"}
    observations = {run.get("observation_sha256") for run in runs if run.get("status") == "PROCESS_VALID"}
    reproducible = (
        len(runs) == repeats
        and all(run.get("status") == "PROCESS_VALID" for run in runs)
        and len(mask_hashes) == 1
        and None not in mask_hashes
        and len(observations) == 1
        and None not in observations
    )
    receipt = {
        "schema": "flashpatch-l7-direct-detector-repeats-v1",
        "repeats_required": 3,
        "comparator": "FlashPatch",
        "fair_runtime_protocol": frozen_runtime,
        "fair_runtime_protocol_sha256": _canonical_json_sha256(frozen_runtime) if frozen_runtime is not None else None,
        "runs": runs,
        "status": "PROCESS_REPRODUCIBLE" if reproducible else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": ["independent_gold_receipt_missing", "frozen_public_case_ledger_missing"],
    }
    receipt_path = root / "repeat-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def aggregate_detection_cases(cases: Sequence[Mapping[str, object]], *, comparators: Sequence[str]) -> dict[str, object]:
    """Calculate a diagnostic summary from normalized labels only.

    This helper deliberately has no receipt, parser, decoder-audit or
    independent-gold inputs.  Its output is useful while developing a corpus,
    but it cannot establish that any labels originated in external executions.
    A future receipt-bound L7 case-bundle verifier must produce the sole
    ``SCOREABLE`` aggregate; this function must never mint one.
    """
    if not cases or list(comparators) != list(DIRECT_DETECTOR_POPULATION):
        raise ExternalLeagueError("L7 diagnostic aggregate requires the fixed direct detector population")
    rows: dict[str, list[tuple[str, str]]] = {name: [] for name in comparators}
    repositories: set[str] = set()
    seen: set[str] = set()
    for case in cases:
        case_id, repository, gold, predictions = (case.get(name) for name in ("case_id", "repository_id", "gold", "predictions"))
        if not isinstance(case_id, str) or not case_id or case_id in seen or not isinstance(repository, str) or not repository or gold not in {"SAFE", "HAZARDOUS"} or not isinstance(predictions, Mapping):
            raise ExternalLeagueError("L7 case identity, independent gold, or predictions are invalid")
        seen.add(case_id)
        repositories.add(repository)
        if set(predictions) != set(comparators):
            raise ExternalLeagueError("L7 comparator slate differs between cases")
        for comparator in comparators:
            repeats = predictions[comparator]
            if not isinstance(repeats, list) or len(repeats) != 3 or any(value not in {"SAFE", "HAZARDOUS"} for value in repeats) or len(set(repeats)) != 1:
                raise ExternalLeagueError("L7 requires three identical normalized predictions per comparator and case")
            rows[comparator].append((str(gold), repeats[0]))
    summaries: dict[str, object] = {}
    for comparator, values in rows.items():
        tp = sum(gold == "HAZARDOUS" and predicted == "HAZARDOUS" for gold, predicted in values)
        tn = sum(gold == "SAFE" and predicted == "SAFE" for gold, predicted in values)
        fp = sum(gold == "SAFE" and predicted == "HAZARDOUS" for gold, predicted in values)
        fn = sum(gold == "HAZARDOUS" and predicted == "SAFE" for gold, predicted in values)
        def f1(a: int, b: int, c: int) -> float:
            return 0.0 if 2 * a + b + c == 0 else 2 * a / (2 * a + b + c)
        summaries[comparator] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": 0.0 if tp + fp == 0 else tp / (tp + fp), "recall": 0.0 if tp + fn == 0 else tp / (tp + fn), "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2}
    return {
        "schema": "flashpatch-l7-detection-aggregate-v1",
        "case_count": len(cases),
        "repository_count": len(repositories),
        "repeats_per_case": 3,
        "comparators": summaries,
        "status": "DIAGNOSTIC_UNBOUND",
        "scoreable": False,
        "scoreable_blockers": [
            "receipt_bound_runs_missing",
            "tool_specific_decode_audits_missing",
            "parser_receipts_missing",
            "independent_gold_receipts_missing",
            "frozen_public_case_ledger_missing",
        ],
    }


def materialize_cfr_ffv1(
    frame_npz: Path | str,
    output_root: Path | str,
    *,
    fps: int,
    ffmpeg_binary: Path | str = "ffmpeg",
) -> dict[str, object]:
    """Freeze renderer frames as lossless FFV1 and prove decode equality.

    The supplied video tool is a *materializer*, not a comparator.  Its binary
    hash, exact command, raw input hashes and decoded RGB hash are preserved so
    no later tool can be credited with a different frame slate.
    """
    source = Path(frame_npz).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"external league output already exists: {root}")
    if not source.is_file():
        raise ExternalLeagueError(f"renderer frame artifact is missing: {source}")
    try:
        with np.load(source) as payload:
            frames = np.asarray(payload["frames"])
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as exc:
        raise ExternalLeagueError(f"renderer artifact must contain frames and timestamps: {exc}") from exc
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ExternalLeagueError("renderer artifact must contain non-empty uint8 RGB frames")
    if timestamps.shape != (len(frames),) or not np.all(np.isfinite(timestamps)):
        raise ExternalLeagueError("renderer timestamps must be finite and one per frame")
    _require_cfr_timeline(timestamps, fps)
    ffmpeg = shutil.which(str(ffmpeg_binary)) if Path(str(ffmpeg_binary)).name == str(ffmpeg_binary) else str(ffmpeg_binary)
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise ExternalLeagueError(f"canonical FFV1 materializer is unavailable: {ffmpeg_binary}")
    root.mkdir(parents=True)
    video = root / "canonical.ffv1.mkv"
    width, height = int(frames.shape[2]), int(frames.shape[1])
    encode_command = [
        str(ffmpeg), "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-", "-an",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24", str(video),
    ]
    started = time.monotonic_ns()
    encoded = subprocess.run(encode_command, input=frames.tobytes(), capture_output=True, check=False)
    elapsed_ns = time.monotonic_ns() - started
    (root / "materializer-encode.stdout.bin").write_bytes(encoded.stdout)
    (root / "materializer-encode.stderr.bin").write_bytes(encoded.stderr)
    if encoded.returncode != 0 or not video.is_file() or video.stat().st_size == 0:
        raise ExternalLeagueError("canonical FFV1 materialization failed")
    decode_command = [str(ffmpeg), "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    decoded = subprocess.run(decode_command, capture_output=True, check=False)
    (root / "materializer-decode.stdout.bin").write_bytes(decoded.stdout)
    (root / "materializer-decode.stderr.bin").write_bytes(decoded.stderr)
    if decoded.returncode != 0 or decoded.stdout != frames.tobytes():
        raise ExternalLeagueError("canonical FFV1 decode does not byte-match renderer RGB input")
    manifest = {
        "schema": "flashpatch-external-cfr-materialization-v1",
        "source": {"path": str(source), "sha256": _sha256_file(source)},
        "cfr": {"fps": fps, "frame_count": len(frames), "shape": list(frames.shape), "timestamps_us": (timestamps * 1_000_000).round().astype(np.int64).tolist()},
        "renderer_rgb": {"raw_sha256": _sha256_bytes(frames.tobytes()), "frame_sha256": _frame_hashes(frames)},
        "canonical_video": {"path": video.name, "sha256": _sha256_file(video), "bytes": video.stat().st_size},
        "materializer": {"binary": str(Path(ffmpeg).resolve()), "binary_sha256": _sha256_file(Path(ffmpeg)), "encode_command": encode_command, "decode_command": decode_command, "encode_exit_code": encoded.returncode, "decode_exit_code": decoded.returncode, "encode_wall_time_ns": elapsed_ns},
        "roundtrip": {"decoded_raw_sha256": _sha256_bytes(decoded.stdout), "byte_identical": True},
    }
    receipt = root / "conversion-receipt.json"
    _write_json(receipt, manifest)
    return {**manifest, "receipt": str(receipt)}


def _native_main_case_file(case_root: Path, value: object, *, field: str) -> Path:
    """Resolve one verifier-owned case file without allowing a path escape."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ExternalLeagueError(f"native-main comparator {field} path is invalid")
    candidate = case_root / value
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(case_root)
    except (OSError, ValueError) as exc:
        raise ExternalLeagueError(
            f"native-main comparator {field} path escapes its case bundle"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ExternalLeagueError(f"native-main comparator {field} is not a regular file")
    return resolved


def materialize_native_main_comparator_input(
    case_root: Path | str,
    output_root: Path | str,
    *,
    ffmpeg_binary: Path | str = "ffmpeg",
) -> dict[str, object]:
    """Convert one reopened native-main case into a non-scoreable FFV1 lane.

    This is the only bridge from L7's renderer qualification bundle to an
    external-comparator input.  It deliberately trusts neither a caller FPS
    nor an unverified NPZ path: both are recovered from the sealed case after
    its native-main verifier has reopened every prerequisite.
    """
    # ``l7_verify`` imports Godot capture helpers which in turn use this
    # module's PNG packer.  Keep the qualification boundary lazy so ordinary
    # comparator imports remain acyclic.
    from .l7_verify import L7VerificationFailure, verify_native_main_natural_case_bundle

    root = Path(case_root).resolve()
    try:
        assessment = verify_native_main_natural_case_bundle(root)
    except (L7VerificationFailure, OSError, ValueError) as exc:
        raise ExternalLeagueError("native-main comparator case did not reopen") from exc
    if (
        not isinstance(assessment, Mapping)
        or assessment.get("status") != "NOT_SCOREABLE"
        or assessment.get("scoreable") is not False
        or assessment.get("native_equivalence") != "NOT_ESTABLISHED"
        or assessment.get("external_claim_authorized") is not False
    ):
        raise ExternalLeagueError("native-main comparator assessment contract is invalid")
    try:
        ledger = json.loads((root / "native-main-natural-case.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("native-main comparator ledger is unreadable") from exc
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("native_main"), Mapping):
        raise ExternalLeagueError("native-main comparator ledger is invalid")
    native = ledger["native_main"]
    frame = _native_main_case_file(root, native.get("frame_artifact_path"), field="frame artifact")
    trace = _native_main_case_file(root, native.get("trace_path"), field="trace")
    receipt = _native_main_case_file(root, native.get("execution_receipt_path"), field="execution receipt")
    ledger_path = root / "native-main-natural-case.json"
    if (
        not ledger_path.is_file()
        or not isinstance(native.get("frame_artifact_sha256"), str)
        or not isinstance(native.get("trace_sha256"), str)
        or not isinstance(native.get("execution_receipt_sha256"), str)
        or _sha256_file(frame) != native["frame_artifact_sha256"]
        or _sha256_file(trace) != native["trace_sha256"]
        or _sha256_file(receipt) != native["execution_receipt_sha256"]
    ):
        raise ExternalLeagueError("native-main comparator case hash binding changed")
    try:
        trace_payload = json.loads(trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("native-main comparator trace is unreadable") from exc
    fps = trace_payload.get("fixed_fps") if isinstance(trace_payload, Mapping) else None
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ExternalLeagueError("native-main comparator trace fixed FPS is invalid")
    conversion = materialize_cfr_ffv1(
        frame, output_root, fps=fps, ffmpeg_binary=ffmpeg_binary
    )
    conversion_receipt = Path(str(conversion["receipt"])).resolve()
    bridge = {
        "schema": "flashpatch-l7-native-main-comparator-input-v1",
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "claim_status": "NOT_SCOREABLE",
        "native_equivalence": "NOT_ESTABLISHED",
        "external_claim_authorized": False,
        "case": {
            "case_id": assessment["case_id"],
            "ledger_sha256": _sha256_file(ledger_path),
            "renderer_execution_receipt_sha256": _sha256_file(receipt),
            "frame_artifact_sha256": _sha256_file(frame),
            "renderer_rgb_sha256": assessment["renderer_rgb_sha256"],
            "timestamps_sha256": assessment["timestamps_sha256"],
            "trace_sha256": _sha256_file(trace),
            "fixed_fps": fps,
        },
        "conversion": {
            "receipt": conversion_receipt.name,
            "receipt_sha256": _sha256_file(conversion_receipt),
            "canonical_video": conversion["canonical_video"],
            "renderer_rgb": conversion["renderer_rgb"],
            "roundtrip": conversion["roundtrip"],
        },
    }
    bridge_path = Path(output_root).resolve() / "native-main-comparator-input-receipt.json"
    _write_json(bridge_path, bridge)
    return {**bridge, "receipt": str(bridge_path)}


def execute_comparator(
    spec: ComparatorSpec,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    census_receipt: Path | str | None = None,
    census_artifact_root: Path | str | None = None,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
    scheduled_repeat_ordinal: int | None = None,
    runtime_schedule: Path | str | None = None,
    schedule_slot: int | None = None,
    tooflashy_parity_adapter_receipt: Path | str | None = None,
) -> dict[str, object]:
    """Execute one frozen external tool and preserve process-level evidence.

    ``PROCESS_VALID`` proves only that a receipt-bound FFV1 lane was supplied
    to a process with declared provenance.  It is intentionally not a scored
    detector result: tool-specific decode audits, parsers, repeats and gold
    labels are separate requirements for a future ``SCOREABLE`` result.
    """
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"comparator output already exists: {root}")
    if not video.is_file():
        raise ExternalLeagueError("canonical comparator video is missing")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    schedule_binding = _load_schedule_assignment(
        runtime_schedule,
        schedule_slot=schedule_slot,
        protocol=frozen_runtime,
        comparator=spec.name,
        repeat_ordinal=scheduled_repeat_ordinal,
        input_sha256=_sha256_file(video),
    )
    if not spec.name or not spec.revision or not spec.repository_url or not spec.license:
        raise ExternalLeagueError("comparator provenance is incomplete")
    if spec.mode not in {"detection", "repair"}:
        raise ExternalLeagueError("comparator mode must be detection or repair")
    if spec.name == "TooFlashy_or_EPI_LENS":
        raise ExternalLeagueError("ambiguous comparator identity cannot execute")
    if spec.name in {"EA IRIS", "EA_IRIS"}:
        raise ExternalLeagueError("ambiguous legacy EA IRIS identity cannot execute")
    if spec.name == EA_IRIS_RELEASE_ORACLE_ID:
        raise ExternalLeagueError("EA IRIS release oracle must execute through the pinned release runner")
    if spec.name == EA_IRIS_SOURCE_ADAPTER_ID:
        raise ExternalLeagueError(
            "EA IRIS source frame adapter is an excluded semantic-mismatch baseline and cannot execute as a participant"
        )
    if spec.name == KAYA_DIRECT_PARTICIPANT_ID:
        raise ExternalLeagueError(
            "Kaya is an unscored participant until natural-case parity and fair-repeat runner receipts exist"
        )
    if spec.name in DIRECT_DETECTOR_POPULATION and spec.mode != "detection":
        raise ExternalLeagueError(f"direct detector cannot execute in the repair lane: {spec.name}")
    if spec.name in MITIGATION_POPULATION and spec.mode != "repair":
        raise ExternalLeagueError("FFmpeg vf_photosensitivity is mitigation-only")
    if spec.name in RESERVE_DETECTOR_POPULATION:
        raise ExternalLeagueError("EPI-LENS is UNSCORABLE until a full same-input application runner exists")
    census_entry: dict[str, object] | None = None
    census_path: Path | None = None
    execution_environment: dict[str, str] | None = None
    if spec.name in {*DIRECT_DETECTOR_POPULATION, *MITIGATION_POPULATION}:
        if census_receipt is None or census_artifact_root is None:
            raise ExternalLeagueError(f"frozen comparator requires a validated census receipt: {spec.name}")
        census_entry, census_path = _load_execution_census_entry(
            census_receipt,
            census_artifact_root,
            spec.name,
        )
        for field, actual in (
            ("repository_url", spec.repository_url),
            ("revision", spec.revision),
            ("license", spec.license),
            ("distribution", spec.distribution),
            ("distribution_revision", spec.distribution_revision),
            ("configuration_sha256", spec.configuration_sha256),
            ("environment_sha256", spec.environment_sha256),
        ):
            if census_entry[field] != actual:
                raise ExternalLeagueError(f"comparator execution differs from census provenance: {spec.name}:{field}")
        if spec.name == "TooFlashy" and (
            spec.source_checkout is None
            or str(spec.source_checkout.resolve()) != str(Path(str(census_entry["source_checkout"])).resolve())
            or spec.working_directory is None
            or spec.working_directory.resolve() != spec.source_checkout.resolve()
        ):
            raise ExternalLeagueError("TooFlashy execution must use the census-pinned source checkout as its working directory")
    if spec.raw_output_mode not in {"file", "stdout"}:
        raise ExternalLeagueError("comparator raw_output_mode must be file or stdout")
    if not spec.expected_exit_codes or any(isinstance(code, bool) or not isinstance(code, int) for code in spec.expected_exit_codes):
        raise ExternalLeagueError("comparator expected_exit_codes must be non-empty integers")
    if isinstance(spec.timeout_seconds, bool) or not isinstance(spec.timeout_seconds, int) or spec.timeout_seconds <= 0:
        raise ExternalLeagueError("comparator timeout_seconds must be a positive integer")
    if frozen_runtime is not None and spec.timeout_seconds != frozen_runtime["budget"]["timeout_seconds"]:
        raise ExternalLeagueError("comparator timeout differs from fair runtime budget")
    if spec.working_directory is not None and not spec.working_directory.is_dir():
        raise ExternalLeagueError("comparator working_directory is unavailable")
    checkout = _checkout_provenance(spec)
    if census_entry is not None and spec.name == "TooFlashy" and checkout.get("status") != "VERIFIED":
        raise ExternalLeagueError("TooFlashy source checkout differs from the census revision")
    adapter_gate: dict[str, object] | None = None
    adapter_gate_path: Path | None = None
    if tooflashy_parity_adapter_receipt is not None:
        if spec.name != "TooFlashy":
            raise ExternalLeagueError("TooFlashy parity adapter cannot be attached to another comparator")
        if frozen_runtime is not None:
            raise ExternalLeagueError(
                "TooFlashy G4 adapter does not authorize fair-runtime scoring until its prebuilt environment is schedule-bound"
            )
        adapter_gate_path = Path(tooflashy_parity_adapter_receipt).resolve()
        adapter_gate = verify_tooflashy_parity_adapter(adapter_gate_path, video, conversion)
        if adapter_gate.get("status") != "VERIFIED":
            raise ExternalLeagueError("TooFlashy parity adapter conformance is not verified")
        adapter_census = adapter_gate.get("census_receipt")
        if (
            census_path is None
            or not isinstance(adapter_census, Mapping)
            or adapter_census.get("path") != str(census_path)
            or adapter_census.get("sha256") != _sha256_file(census_path)
            or adapter_gate.get("upstream", {}).get("revision") != spec.revision
            or adapter_gate.get("upstream", {}).get("path") != str(spec.source_checkout.resolve())
        ):
            raise ExternalLeagueError("TooFlashy parity adapter differs from comparator census or source")
    raw_output = root / "raw-output.bin"
    command = [part.format(input=str(video), output=str(raw_output)) for part in spec.command]
    if not command or any("{" in part or "}" in part for part in command):
        raise ExternalLeagueError("comparator command contains an unresolved placeholder")
    executable = shutil.which(command[0]) if Path(command[0]).name == command[0] else command[0]
    if not executable or not Path(executable).is_file():
        raise ExternalLeagueError(f"comparator executable unavailable: {command[0]}")
    command[0] = str(Path(executable).resolve())
    if census_entry is not None:
        if _sha256_file(Path(command[0])) != census_entry["binary_sha256"]:
            raise ExternalLeagueError(f"comparator execution binary differs from census: {spec.name}")
        command_artifact = _resolve_census_artifact(
            Path(census_artifact_root).resolve(),
            census_entry["command_artifact"],
            name=spec.name,
            field="command_artifact",
        )
        try:
            frozen_command = json.loads(command_artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError(f"comparator census command is unreadable: {spec.name}") from exc
        if frozen_command != list(spec.command):
            raise ExternalLeagueError(f"comparator command template differs from census: {spec.name}")
        environment_artifact = _resolve_census_artifact(
            Path(census_artifact_root).resolve(),
            census_entry["environment_artifact"],
            name=spec.name,
            field="environment_artifact",
        )
        try:
            frozen_environment = json.loads(environment_artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError(f"comparator census environment is unreadable: {spec.name}") from exc
        if not isinstance(frozen_environment, Mapping) or not all(isinstance(key, str) and isinstance(value, str) for key, value in frozen_environment.items()):
            raise ExternalLeagueError(f"comparator census environment is invalid: {spec.name}")
        execution_environment = dict(frozen_environment)
    root.mkdir(parents=True)
    comparator_binary = Path(command[0]).resolve()
    using_tooflashy_adapter = adapter_gate is not None
    adapter_build: dict[str, object] | None = None
    adapter_witness_path: Path | None = None
    if using_tooflashy_adapter:
        adapter_environment = {
            "HOME": str(root / "home"),
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "UV_PROJECT": str(spec.source_checkout.resolve()),
            "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
        }
        (root / "home").mkdir()
        build_command = [
            str(comparator_binary), "sync", "--locked", "--project",
            str(spec.source_checkout.resolve()), "--directory",
            str(spec.source_checkout.resolve()),
        ]
        build = subprocess.run(
            build_command,
            capture_output=True,
            check=False,
            timeout=spec.timeout_seconds,
            cwd=spec.source_checkout,
            env=adapter_environment,
        )
        (root / "adapter-build.stdout.bin").write_bytes(build.stdout)
        (root / "adapter-build.stderr.bin").write_bytes(build.stderr)
        adapter_build = {
            "command": build_command,
            "environment": adapter_environment,
            "environment_sha256": _canonical_json_sha256(adapter_environment),
            "exit_code": build.returncode,
            "stdout_sha256": _sha256_bytes(build.stdout),
            "stderr_sha256": _sha256_bytes(build.stderr),
        }
        if build.returncode != 0:
            raise ExternalLeagueError("TooFlashy parity adapter dependency build failed")
        adapter_hash = _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
        env_binary = Path("/usr/bin/env").resolve()
        adapter_tool_command = [
            str(env_binary), "-i",
            *[f"{key}={value}" for key, value in sorted(adapter_environment.items())],
            str(comparator_binary), "run", "--locked", "--no-sync", "--project",
            str(spec.source_checkout.resolve()), "--directory",
            str(spec.source_checkout.resolve()), "python", "-c",
            _TOOFLASHY_PARITY_ADAPTER_SCRIPT, str(video), str(raw_output), adapter_hash,
        ]
        adapter_witness_path = root / "runtime-probe.json"
        tool_command = adapter_tool_command
    else:
        tool_command = list(command)
    with _fair_execution_context(
        frozen_runtime,
        video,
        base_environment=execution_environment,
        schedule_binding=schedule_binding,
        launcher_cwd=spec.working_directory,
    ) as execution:
        probe_path = root / "runtime-probe.json"
        command = _instrument_fair_command(execution, tool_command, probe_path)
        started = time.monotonic_ns()
        conversion_payload = _load_conversion_receipt(conversion, video)
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=spec.timeout_seconds,
                cwd=str(spec.working_directory) if spec.working_directory is not None else None,
                env=execution["environment"],
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: comparator timeout",
            )
        if spec.raw_output_mode == "stdout" and not using_tooflashy_adapter:
            raw_output.write_bytes(completed.stdout)
        (root / "stdout.bin").write_bytes(completed.stdout)
        (root / "stderr.bin").write_bytes(completed.stderr)
        process_valid = (
            completed.returncode in spec.expected_exit_codes
            and raw_output.is_file()
            and raw_output.stat().st_size > 0
            and not timed_out
        )
        normalized_observation: dict[str, object] | None = None
        parse_error: str | None = None
        normalizer = "UNAVAILABLE"
        if process_valid and spec.name == "TooFlashy":
            normalizer = (
                "tooflashy-adapter-json-v1"
                if using_tooflashy_adapter
                else "tooflashy-json-v1"
            )
            cfr = conversion_payload.get("cfr")
            try:
                if not isinstance(cfr, Mapping):
                    raise ExternalLeagueError("canonical conversion receipt omits CFR metadata")
                fps = cfr.get("fps")
                frame_count = cfr.get("frame_count")
                if isinstance(fps, bool) or not isinstance(fps, int) or isinstance(frame_count, bool) or not isinstance(frame_count, int):
                    raise ExternalLeagueError("canonical conversion receipt CFR metadata is invalid")
                parser = (
                    parse_tooflashy_adapter_json
                    if using_tooflashy_adapter
                    else parse_tooflashy_json
                )
                normalized_observation = parser(
                    raw_output, video, expected_fps=fps, expected_frame_count=frame_count
                )
            except ExternalLeagueError as exc:
                parse_error = str(exc)
        child_probe: dict[str, object] | None = None
        if frozen_runtime is not None:
            try:
                child_probe = _load_child_runtime_probe(probe_path)
            except ExternalLeagueError:
                pass
        adapter_witness: dict[str, object] | None = None
        if using_tooflashy_adapter and adapter_witness_path is not None:
            try:
                adapter_witness = _load_child_runtime_probe(adapter_witness_path)
            except ExternalLeagueError:
                pass
        finished = time.monotonic_ns()
        elapsed_ns = finished - started
        if frozen_runtime is not None and elapsed_ns > int(frozen_runtime["budget"]["timeout_seconds"]) * 1_000_000_000:
            timed_out = True
        if timed_out:
            process_valid = False
        if spec.name == "TooFlashy" and normalized_observation is None:
            process_valid = False
        if frozen_runtime is not None and normalized_observation is None:
            process_valid = False
        if frozen_runtime is not None and child_probe is None:
            process_valid = False
        if using_tooflashy_adapter and adapter_witness is None:
            process_valid = False
        runtime_receipt = _fair_runtime_run_receipt(
            frozen_runtime,
            comparator=spec.name,
            scheduled_repeat_ordinal=scheduled_repeat_ordinal,
            schedule_binding=schedule_binding,
            input_sha256=_sha256_file(video),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            wall_time_ns=elapsed_ns,
            timed_out=timed_out,
            observation=normalized_observation,
            normalizer=normalizer,
            observed_environment={
                "parent_precondition": execution["observation"],
                "child_probe": child_probe,
            } if frozen_runtime is not None else None,
        )
    receipt = {
        "schema": "flashpatch-external-comparator-run-v1",
        "comparator": {"name": spec.name, "repository_url": spec.repository_url, "revision": spec.revision, "license": spec.license, "mode": spec.mode, "binary": str(comparator_binary), "binary_sha256": _sha256_file(comparator_binary), "working_directory": str(spec.working_directory.resolve()) if spec.working_directory is not None else None, "expected_exit_codes": list(spec.expected_exit_codes), "source_checkout": checkout},
        "input": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion), "renderer_rgb_sha256": conversion_payload["renderer_rgb"]["raw_sha256"]},
        "census_receipt": {"path": str(census_path), "sha256": _sha256_file(census_path)} if census_path is not None else None,
        "command": command,
        "exit_code": completed.returncode,
        "wall_time_ns": elapsed_ns,
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stderr_sha256": _sha256_bytes(completed.stderr),
        "raw_output": {"path": raw_output.name, "mode": "file" if using_tooflashy_adapter else spec.raw_output_mode, "exists": raw_output.is_file(), "sha256": _sha256_file(raw_output) if raw_output.is_file() else None},
        "tooflashy_parity_adapter": {
            "path": str(adapter_gate_path),
            "sha256": _sha256_file(adapter_gate_path),
            "status": adapter_gate.get("status"),
        } if adapter_gate_path is not None and adapter_gate is not None else None,
        "adapter_build": adapter_build,
        "adapter_execution_witness": {
            "path": adapter_witness_path.name,
            "sha256": _sha256_file(adapter_witness_path)
            if adapter_witness_path is not None and adapter_witness_path.is_file()
            else None,
            "observation": adapter_witness,
        } if adapter_witness_path is not None else None,
        "parsed_observation": normalized_observation,
        "parse_error": parse_error,
        "fair_runtime": runtime_receipt,
        "runtime_probe": {
            "path": probe_path.name,
            "sha256": _sha256_file(probe_path) if frozen_runtime is not None and probe_path.is_file() else None,
            "observation": child_probe if frozen_runtime is not None else None,
        },
        "status": "PROCESS_VALID" if process_valid else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": (
            ["runtime_timeout"]
            if timed_out
            else [
                *([] if using_tooflashy_adapter else ["tool_specific_input_decode_audit_missing"]),
                *(["tool_specific_prediction_parser_missing"] if normalized_observation is None else []),
                "three_repeat_and_retry_ledger_missing",
                "independent_gold_oracle_missing",
            ]
        ),
    }
    receipt_path = root / "comparator-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def execute_repeated_comparator(
    spec: ComparatorSpec,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    repeats: int = 3,
    census_receipt: Path | str | None = None,
    census_artifact_root: Path | str | None = None,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object] | None = None,
    tooflashy_parity_adapter_receipt: Path | str | None = None,
) -> dict[str, object]:
    """Run a fixed comparator repeatedly without turning retry into a win.

    Repeats are independent output roots. Any incomplete run remains visible in
    the ledger and prevents process reproducibility; no result is retried or
    dropped silently.
    """
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"comparator repeat output already exists: {root}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 3:
        raise ExternalLeagueError("L7 requires exactly three comparator repeats")
    frozen_runtime = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen_runtime is not None and frozen_runtime["budget"]["scheduled_repeats"] != repeats:
        raise ExternalLeagueError("comparator repeat count differs from fair runtime budget")
    root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    for index in range(repeats):
        try:
            run = execute_comparator(
                spec,
                canonical_video,
                conversion_receipt,
                root / f"run-{index + 1}",
                census_receipt=census_receipt,
                census_artifact_root=census_artifact_root,
                runtime_protocol=frozen_runtime,
                scheduled_repeat_ordinal=index + 1 if frozen_runtime is not None else None,
                tooflashy_parity_adapter_receipt=tooflashy_parity_adapter_receipt,
            )
            observation = run.get("parsed_observation")
            runs.append({
                "repeat": index + 1,
                "status": run["status"],
                "receipt": run["receipt"],
                "receipt_sha256": _sha256_file(Path(str(run["receipt"]))),
                "raw_output_sha256": run["raw_output"]["sha256"],
                "normalized_observation_sha256": _canonical_json_sha256(observation) if isinstance(observation, Mapping) else None,
                "fair_runtime": run.get("fair_runtime"),
                "exit_code": run["exit_code"],
            })
        except (ExternalLeagueError, OSError, subprocess.SubprocessError) as exc:
            runs.append({"repeat": index + 1, "status": "INCONCLUSIVE", "reason": str(exc)})
    observations = {run.get("normalized_observation_sha256") for run in runs if run.get("status") == "PROCESS_VALID"}
    reproducible = (
        len(runs) == repeats
        and all(run.get("status") == "PROCESS_VALID" for run in runs)
        and len(observations) == 1
        and None not in observations
    )
    receipt = {
        "schema": "flashpatch-external-comparator-repeats-v1",
        "repeats_required": 3,
        "comparator": spec.name,
        "fair_runtime_protocol": frozen_runtime,
        "fair_runtime_protocol_sha256": _canonical_json_sha256(frozen_runtime) if frozen_runtime is not None else None,
        "runs": runs,
        "status": "PROCESS_REPRODUCIBLE" if reproducible else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": [
            *(["normalized_terminal_observation_missing"] if not reproducible else []),
            "tool_specific_input_decode_audit_missing",
            "independent_gold_oracle_missing",
        ],
    }
    receipt_path = root / "repeat-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def _reopen_child_normalized_observation(
    child_path: Path,
    child: Mapping[str, object],
    comparator: str,
    input_sha256: str,
) -> dict[str, object]:
    root = child_path.parent
    schema_by_comparator = {
        "FlashPatch": "flashpatch-l7-direct-detector-run-v1",
        KAYA_DIRECT_PARTICIPANT_ID: KAYA_FAIR_RUNTIME_RUN_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-run-v1",
    }
    if child.get("schema") != schema_by_comparator.get(comparator) or child.get("status") != "PROCESS_VALID":
        raise ExternalLeagueError("child run schema or terminal status mismatches comparator")
    comparator_payload = child.get("comparator")
    if not isinstance(comparator_payload, Mapping) or comparator_payload.get("name") != comparator:
        raise ExternalLeagueError("child run comparator identity mismatches repeat receipt")
    conversion_ref = child.get("conversion_receipt")
    if not isinstance(conversion_ref, Mapping) or not isinstance(conversion_ref.get("path"), str):
        raise ExternalLeagueError("child run does not bind its conversion receipt path")
    conversion = Path(str(conversion_ref["path"])).resolve()
    if not conversion.is_file() or conversion_ref.get("sha256") != _sha256_file(conversion):
        raise ExternalLeagueError("child conversion receipt hash mismatches")

    def reopen_census() -> dict[str, object]:
        census_ref = child.get("census_receipt")
        if not isinstance(census_ref, Mapping) or not isinstance(census_ref.get("path"), str):
            raise ExternalLeagueError(f"{comparator} child census receipt is missing")
        census = Path(str(census_ref["path"])).resolve()
        if not census.is_file() or census_ref.get("sha256") != _sha256_file(census):
            raise ExternalLeagueError(f"{comparator} child census receipt hash mismatches")
        try:
            census_payload = json.loads(census.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError(f"{comparator} child census receipt is unreadable") from exc
        artifact_root = census_payload.get("artifact_root") if isinstance(census_payload, Mapping) else None
        if not isinstance(artifact_root, str):
            raise ExternalLeagueError(f"{comparator} child census artifact root is missing")
        entry, _ = _load_execution_census_entry(census, artifact_root, comparator)
        return entry

    if comparator == KAYA_DIRECT_PARTICIPANT_ID:
        reopened = verify_kaya_scheduled_fair_runtime_run_receipt(child_path)
        input_payload = reopened.get("input")
        observation = reopened.get("observation")
        if (
            not isinstance(input_payload, Mapping)
            or input_payload.get("sha256") != input_sha256
            or not isinstance(observation, Mapping)
        ):
            raise ExternalLeagueError("Kaya fair-runtime input or observation is invalid")
        return dict(observation)

    if comparator == EA_IRIS_SOURCE_ADAPTER_ID:
        reopened, _, observation, reopened_path = _load_iris_source_adapter_run(
            child_path,
            expected_lane="DIRECT_DETECTOR",
        )
        input_payload = reopened.get("input")
        if (
            reopened_path != child_path.resolve()
            or not isinstance(input_payload, Mapping)
            or input_payload.get("sha256") != input_sha256
            or _sha256_file(Path(str(input_payload.get("path", ""))).resolve()) != input_sha256
        ):
            raise ExternalLeagueError("EA IRIS source adapter input differs from fair runtime input")
        census_entry = reopen_census()
        build_ref = reopened.get("build_receipt")
        if not isinstance(build_ref, Mapping) or not isinstance(build_ref.get("path"), str):
            raise ExternalLeagueError("EA IRIS source adapter build receipt is missing")
        build, _ = _load_iris_source_build_receipt(build_ref["path"])
        if (
            census_entry.get("repository_url") != build["source"]["repository_url"]
            or census_entry.get("revision") != build["source"]["revision"]
            or census_entry.get("license") != "BSD-3-Clause"
        ):
            raise ExternalLeagueError("EA IRIS source adapter census differs from frozen build")
        return observation
    input_payload = child.get("input")
    if not isinstance(input_payload, Mapping):
        raise ExternalLeagueError("child run input identity is missing")
    video_path = (
        Path(str(input_payload.get("path", ""))).resolve()
        if comparator == "TooFlashy"
        else (conversion.parent / "canonical.ffv1.mkv").resolve()
    )
    video_hash = input_payload.get("sha256") if comparator == "TooFlashy" else input_payload.get("canonical_video_sha256")
    if not video_path.is_file() or video_hash != input_sha256 or _sha256_file(video_path) != input_sha256:
        raise ExternalLeagueError("child canonical input identity mismatches fair runtime")
    conversion_payload = _load_conversion_receipt(conversion, video_path)
    if comparator == "TooFlashy":
        census_entry = reopen_census()
        for field in ("repository_url", "revision", "license", "binary_sha256"):
            if comparator_payload.get(field) != census_entry.get(field):
                raise ExternalLeagueError(f"TooFlashy child provenance differs from census: {field}")
        raw = child.get("raw_output")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ExternalLeagueError("TooFlashy child raw output is missing")
        raw_path = _resolve_run_owned_artifact(
            child_path,
            raw["path"],
            label="TooFlashy child raw output",
        )
        if not raw_path.is_file() or raw.get("sha256") != _sha256_file(raw_path):
            raise ExternalLeagueError("TooFlashy child raw output hash mismatches")
        cfr = conversion_payload.get("cfr")
        if not isinstance(cfr, Mapping):
            raise ExternalLeagueError("TooFlashy conversion receipt omits CFR metadata")
        try:
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError("TooFlashy child raw output cannot be reopened") from exc
        parser = (
            parse_tooflashy_adapter_json
            if isinstance(raw_payload, Mapping)
            and raw_payload.get("schema") == "flashpatch-l7-tooflashy-child-adapter-v1"
            else parse_tooflashy_json
        )
        return parser(
            raw_path,
            video_path,
            expected_fps=int(cfr["fps"]),
            expected_frame_count=int(cfr["frame_count"]),
        )
    if comparator != "FlashPatch":
        raise ExternalLeagueError("unsupported fair runtime comparator")
    mask = child.get("hazard_mask")
    decode = child.get("worker_decode")
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str) or not isinstance(decode, Mapping) or not isinstance(decode.get("path"), str):
        raise ExternalLeagueError("FlashPatch child raw detector evidence is missing")
    mask_path = _resolve_run_owned_artifact(
        child_path,
        mask["path"],
        label="FlashPatch child hazard mask",
    )
    decode_path = _resolve_run_owned_artifact(
        child_path,
        decode["path"],
        label="FlashPatch child decoder audit",
    )
    if (
        not mask_path.is_file()
        or mask.get("sha256") != _sha256_file(mask_path)
        or not decode_path.is_file()
        or decode.get("sha256") != _sha256_file(decode_path)
    ):
        raise ExternalLeagueError("FlashPatch child raw detector evidence hash mismatches")
    frames, timestamps, decode_receipt = _decode_canonical_video_rgb(video_path, conversion)
    if decode.get("receipt") != decode_receipt:
        raise ExternalLeagueError("FlashPatch child decoder observation mismatches canonical replay")
    result = analyze(frames, timestamps)
    expected_mask = result.hazard_mask
    try:
        stored_mask = np.load(mask_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ExternalLeagueError("FlashPatch hazard mask is unreadable") from exc
    if stored_mask.shape != expected_mask.shape or not np.array_equal(stored_mask, expected_mask):
        raise ExternalLeagueError("FlashPatch hazard mask mismatches canonical detector replay")
    return {
        "tool": "FlashPatch",
        "prediction": "HAZARDOUS" if result.hazardous else "SAFE",
        "frame_count": len(frames),
        "fps": decode_receipt["fps"],
        "hazard_frame_indices": np.flatnonzero(np.any(result.hazard_mask, axis=(1, 2))).astype(int).tolist(),
        "max_flash_count": result.max_flash_count,
        "max_affected_fraction": result.max_affected_fraction,
        "hazard_kinds": sorted(result.kind_masks),
        "timestamp_metrics": "canonical_cfr_timestamp",
        "mask_metrics": "hazard_mask_npy",
    }


def _canonical_decoder_timeline_contract(
    video: Path,
    conversion: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Reopen the renderer source and lossless conversion as one frame map."""
    payload = _load_conversion_receipt(conversion, video)
    source_ref = payload.get("source")
    cfr = payload.get("cfr")
    renderer = payload.get("renderer_rgb")
    if not all(isinstance(item, Mapping) for item in (source_ref, cfr, renderer)):
        raise ExternalLeagueError("canonical decoder contract omits renderer provenance")
    source = Path(str(source_ref.get("path", ""))).resolve()
    if not source.is_file() or source_ref.get("sha256") != _sha256_file(source):
        raise ExternalLeagueError("canonical renderer source hash mismatches conversion receipt")
    try:
        with np.load(source) as archive:
            frames = np.asarray(archive["frames"])
            renderer_timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as exc:
        raise ExternalLeagueError("canonical renderer source cannot be reopened") from exc
    fps = cfr.get("fps")
    frame_count = cfr.get("frame_count")
    shape = cfr.get("shape")
    if (
        frames.ndim != 4
        or frames.dtype != np.uint8
        or frames.shape[-1] != 3
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or fps <= 0
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != len(frames)
        or shape != list(frames.shape)
        or renderer_timestamps.shape != (frame_count,)
        or not np.all(np.isfinite(renderer_timestamps))
    ):
        raise ExternalLeagueError("canonical renderer frame or CFR shape is invalid")
    _require_cfr_timeline(renderer_timestamps, fps)
    renderer_timestamps_us = (renderer_timestamps * 1_000_000).round().astype(np.int64).tolist()
    cfr_timestamps_us = (
        np.arange(frame_count, dtype=np.float64) * 1_000_000 / float(fps)
    ).round().astype(np.int64).tolist()
    frame_hashes = _frame_hashes(frames)
    if (
        cfr.get("timestamps_us") != renderer_timestamps_us
        or renderer.get("raw_sha256") != _sha256_bytes(frames.tobytes())
        or renderer.get("frame_sha256") != frame_hashes
    ):
        raise ExternalLeagueError("canonical renderer frame map drifts from conversion receipt")
    decoded_frames, _, decode_receipt = _decode_canonical_video_rgb(video, conversion)
    if not np.array_equal(decoded_frames, frames):
        raise ExternalLeagueError("canonical FFV1 fresh decode differs from renderer frames")
    frame_map = [
        {
            "frame_index": index,
            "cfr_timestamp_us": int(cfr_timestamps_us[index]),
            "renderer_timestamp_us": int(renderer_timestamps_us[index]),
            "rgb_sha256": frame_hashes[index],
        }
        for index in range(frame_count)
    ]
    contract = {
        "canonical_video": {
            "path": str(video),
            "sha256": _sha256_file(video),
        },
        "conversion_receipt": {
            "path": str(conversion),
            "sha256": _sha256_file(conversion),
        },
        "renderer_source": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "rgb_sha256": renderer["raw_sha256"],
        },
        "fps": fps,
        "frame_count": frame_count,
        "shape": shape,
        "frame_map": frame_map,
        "frame_map_sha256": _canonical_json_sha256(frame_map),
        "fresh_ffv1_decode": decode_receipt,
    }
    return payload, contract


def _parse_timestamp_us(value: object) -> int:
    if not isinstance(value, str):
        raise ExternalLeagueError("native timestamp is not a string")
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})\.(\d{6})", value)
    if match is None:
        raise ExternalLeagueError("native timestamp format is invalid")
    hours, minutes, seconds, micros = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ExternalLeagueError("native timestamp clock value is invalid")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000_000 + micros


def _resolve_run_owned_artifact(child_path: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise ExternalLeagueError(f"{label} path is not a run-owned relative artifact")
    root = child_path.parent.resolve()
    artifact = (root / relative_path).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise ExternalLeagueError(f"{label} path escapes the child run root") from exc
    if artifact == root:
        raise ExternalLeagueError(f"{label} path does not identify a file")
    return artifact


def _audit_iris_decoder_timeline(
    child_path: Path,
    child: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    frame_report = child.get("frame_report")
    if not isinstance(frame_report, Mapping) or not isinstance(frame_report.get("path"), str):
        raise ExternalLeagueError("EA IRIS native frame CSV is missing")
    csv_path = _resolve_run_owned_artifact(
        child_path,
        frame_report["path"],
        label="EA IRIS native frame CSV",
    )
    stdout_path = _resolve_run_owned_artifact(
        child_path,
        "stdout.bin",
        label="EA IRIS stdout",
    )
    staged_video = child_path.parent / "TestVideos" / "canonical.ffv1.mkv"
    if (
        not csv_path.is_file()
        or frame_report.get("sha256") != _sha256_file(csv_path)
        or not stdout_path.is_file()
        or child.get("stdout_sha256") != _sha256_file(stdout_path)
        or not staged_video.is_file()
        or _sha256_file(staged_video) != contract["canonical_video"]["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS native decode artifacts or staged input hash mismatch")
    try:
        raw_csv = csv_path.read_bytes().replace(b"\x00", b"")
        rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ExternalLeagueError("EA IRIS native frame CSV cannot be reopened") from exc
    frame_count = int(contract["frame_count"])
    if len(rows) != frame_count:
        raise ExternalLeagueError("EA IRIS native frame count differs from canonical timeline")
    try:
        native_frames = [int(row["Frame"]) for row in rows]
        native_timestamps_us = [_parse_timestamp_us(row["TimeStamp"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalLeagueError("EA IRIS native frame map is invalid") from exc
    if native_frames != list(range(1, frame_count + 1)):
        raise ExternalLeagueError("EA IRIS native frame map drops, duplicates, or reorders frames")
    if frame_count > 1 and any(
        later <= earlier
        for earlier, later in zip(native_timestamps_us, native_timestamps_us[1:])
    ):
        raise ExternalLeagueError("EA IRIS native timestamps are not strictly increasing")
    canonical_times = [int(row["cfr_timestamp_us"]) for row in contract["frame_map"]]
    errors = [abs(actual - expected) for actual, expected in zip(native_timestamps_us, canonical_times)]
    if any(error > EA_IRIS_RELEASE_TIMESTAMP_PRECISION_US for error in errors):
        raise ExternalLeagueError("EA IRIS timestamp drift exceeds its declared millisecond precision")
    return {
        "parity_status": "NOT_VERIFIED",
        "parity_reason": "native_csv_omits_decoded_rgb_frame_identity",
        "native_artifacts": [
            {"path": str(csv_path), "sha256": _sha256_file(csv_path)},
            {"path": str(stdout_path), "sha256": _sha256_file(stdout_path)},
        ],
        "actual_input": {"path": str(staged_video), "sha256": _sha256_file(staged_video)},
        "reported_frame_count": frame_count,
        "fps": contract["fps"],
        "frame_index_base": 1,
        "timestamp_precision_us": EA_IRIS_RELEASE_TIMESTAMP_PRECISION_US,
        "max_timestamp_alignment_error_us": max(errors, default=0),
        "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
        "canonical_frame_map_sha256": contract["frame_map_sha256"],
        "frame_map_relation": "SEQUENTIAL_ONE_BASED_WITH_DECLARED_TIMESTAMP_PRECISION",
        "native_frame_map_sha256": _canonical_json_sha256(
            [
                {"native_frame": frame, "timestamp_us": timestamp}
                for frame, timestamp in zip(native_frames, native_timestamps_us)
            ]
        ),
    }


def _verify_tooflashy_adapter_checkout(entry: Mapping[str, object]) -> dict[str, object]:
    provenance = _verify_upstream_checkout(entry)
    checkout = Path(str(provenance.get("path", ""))).resolve()
    full_status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        check=False,
    )
    if (
        provenance.get("revision") != TOOFLASHY_PARITY_ADAPTER_REVISION
        or provenance.get("tree") != TOOFLASHY_PARITY_ADAPTER_TREE
        or provenance.get("origin") != "https://github.com/hashb/TooFlashy"
        or provenance.get("clean") is not True
        or full_status.returncode != 0
        or full_status.stdout != b""
    ):
        raise ExternalLeagueError("TooFlashy adapter checkout is not the frozen upstream tree")
    observed: dict[str, str] = {}
    for relative, expected in TOOFLASHY_PARITY_SOURCE_HASHES.items():
        path = (checkout / relative).resolve()
        try:
            path.relative_to(checkout)
        except ValueError as exc:
            raise ExternalLeagueError("TooFlashy adapter source path escapes checkout") from exc
        if not path.is_file():
            raise ExternalLeagueError(f"TooFlashy adapter source is missing: {relative}")
        observed[relative] = _sha256_file(path)
        if observed[relative] != expected:
            raise ExternalLeagueError(f"TooFlashy adapter source drifted: {relative}")
    if entry.get("license_sha256") != observed["LICENSE"]:
        raise ExternalLeagueError("TooFlashy adapter license differs from census")
    return {
        "path": str(checkout),
        "repository_url": "https://github.com/hashb/TooFlashy",
        "revision": TOOFLASHY_PARITY_ADAPTER_REVISION,
        "tree": TOOFLASHY_PARITY_ADAPTER_TREE,
        "license": "Apache-2.0",
        "source_sha256": observed,
    }


def _tooflashy_adapter_process_receipt(
    completed: subprocess.CompletedProcess[bytes],
    *,
    command: Sequence[str],
    working_directory: Path,
    environment: Mapping[str, str],
    root: Path,
    prefix: str,
    timed_out: bool,
) -> dict[str, object]:
    stdout = root / f"{prefix}.stdout.bin"
    stderr = root / f"{prefix}.stderr.bin"
    stdout.write_bytes(completed.stdout or b"")
    stderr.write_bytes(completed.stderr or b"")
    return {
        "command": list(command),
        "working_directory": str(working_directory),
        "environment_sha256": _canonical_json_sha256(dict(environment)),
        "exit_code": completed.returncode,
        "timed_out": timed_out,
        "stdout": {"path": stdout.name, "sha256": _sha256_file(stdout)},
        "stderr": {"path": stderr.name, "sha256": _sha256_file(stderr)},
    }


def _load_tooflashy_conformance_manifest(path: Path | str) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("TooFlashy conformance fixture manifest is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema", "fixtures"}
        or payload.get("schema") != TOOFLASHY_CONFORMANCE_FIXTURES_SCHEMA
        or not isinstance(payload.get("fixtures"), list)
        or not payload["fixtures"]
    ):
        raise ExternalLeagueError("TooFlashy conformance fixture manifest fields are invalid")
    fixtures: list[dict[str, object]] = []
    identifiers: list[str] = []
    hashes: list[str] = []
    for row in payload["fixtures"]:
        if not isinstance(row, Mapping) or set(row) != {"id", "path", "sha256", "bytes"}:
            raise ExternalLeagueError("TooFlashy conformance fixture entry fields are invalid")
        identifier = row.get("id")
        relative = row.get("path")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", identifier) is None
            or not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ExternalLeagueError("TooFlashy conformance fixture identity or path is invalid")
        fixture = (manifest_path.parent / relative).resolve()
        try:
            fixture.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ExternalLeagueError("TooFlashy conformance fixture escapes manifest root") from exc
        if (
            not fixture.is_file()
            or row.get("sha256") != _sha256_file(fixture)
            or row.get("bytes") != fixture.stat().st_size
        ):
            raise ExternalLeagueError("TooFlashy conformance fixture hash or size mismatches")
        identifiers.append(identifier)
        hashes.append(str(row["sha256"]))
        fixtures.append({
            "id": identifier,
            "path": str(fixture),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        })
    if len(identifiers) != len(set(identifiers)) or len(hashes) != len(set(hashes)):
        raise ExternalLeagueError("TooFlashy conformance fixtures must have unique identities and content")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "fixtures": fixtures,
    }


def execute_tooflashy_parity_adapter(
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    *,
    census_receipt: Path | str,
    census_artifact_root: Path | str,
    conformance_manifest: Path | str,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    """Run the pinned public TooFlashy APIs and its official CLI symmetrically.

    This creates decode-parity evidence only.  Even a verified result is not a
    detector score, a gold label, a rank, or a winner claim.
    """
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"TooFlashy adapter output already exists: {root}")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ExternalLeagueError("TooFlashy adapter timeout must be a positive integer")
    _, canonical_contract = _canonical_decoder_timeline_contract(video, conversion)
    entry, census_path = _load_execution_census_entry(
        census_receipt,
        census_artifact_root,
        "TooFlashy",
    )
    upstream = _verify_tooflashy_adapter_checkout(entry)
    checkout = Path(str(upstream["path"]))
    uv = _uv_executable()
    if not uv.is_file() or _sha256_file(uv) != entry.get("binary_sha256"):
        raise ExternalLeagueError("TooFlashy adapter uv executable differs from census")
    frozen_fixtures = _load_tooflashy_conformance_manifest(conformance_manifest)
    extra_videos = [Path(str(item["path"])).resolve() for item in frozen_fixtures["fixtures"]]
    videos = [video, *extra_videos]
    video_hashes = [_sha256_file(item) for item in videos]
    if (
        any(not item.is_file() for item in videos)
        or len(set(videos)) != len(videos)
        or len(set(video_hashes)) != len(video_hashes)
    ):
        raise ExternalLeagueError("TooFlashy adapter fixture videos must exist and have unique paths and content")

    root.mkdir(parents=True)
    environment = {
        "HOME": str(root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_PROJECT": str(checkout),
        "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
    }
    (root / "home").mkdir()
    adapter_hash = _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
    fixture_rows: list[dict[str, object]] = []
    fixture_ids = ["canonical", *[str(item["id"]) for item in frozen_fixtures["fixtures"]]]
    for ordinal, fixture in enumerate(videos):
        fixture_root = root / f"fixture-{ordinal:03d}"
        fixture_root.mkdir()
        adapter_output = fixture_root / "adapter-output.json"
        adapter_command = [
            str(uv),
            "run",
            "--locked",
            "--project",
            str(checkout),
            "--directory",
            str(checkout),
            "python",
            "-c",
            _TOOFLASHY_PARITY_ADAPTER_SCRIPT,
            str(fixture),
            str(adapter_output),
            adapter_hash,
        ]
        adapter_timed_out = False
        try:
            adapter_completed = subprocess.run(
                adapter_command,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=checkout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            adapter_timed_out = True
            adapter_completed = subprocess.CompletedProcess(
                adapter_command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: TooFlashy adapter timeout",
            )
        cli_command = [
            str(uv),
            "run",
            "--locked",
            "--project",
            str(checkout),
            "--directory",
            str(checkout),
            "tooflashy",
            "--json",
            str(fixture),
        ]
        cli_timed_out = False
        try:
            cli_completed = subprocess.run(
                cli_command,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=checkout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            cli_timed_out = True
            cli_completed = subprocess.CompletedProcess(
                cli_command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: TooFlashy CLI timeout",
            )
        cli_output = fixture_root / "official-cli-output.json"
        cli_output.write_bytes(cli_completed.stdout or b"")
        fixture_rows.append({
            "ordinal": ordinal,
            "fixture_id": fixture_ids[ordinal],
            "role": "CANONICAL" if ordinal == 0 else "CONFORMANCE",
            "input": {
                "path": str(fixture),
                "sha256": _sha256_file(fixture),
                "bytes": fixture.stat().st_size,
            },
            "adapter_process": _tooflashy_adapter_process_receipt(
                adapter_completed,
                command=adapter_command,
                working_directory=checkout,
                environment=environment,
                root=fixture_root,
                prefix="adapter",
                timed_out=adapter_timed_out,
            ),
            "adapter_output": {
                "path": str(adapter_output.relative_to(root)),
                "exists": adapter_output.is_file(),
                "sha256": _sha256_file(adapter_output) if adapter_output.is_file() else None,
            },
            "official_cli_process": _tooflashy_adapter_process_receipt(
                cli_completed,
                command=cli_command,
                working_directory=checkout,
                environment=environment,
                root=fixture_root,
                prefix="official-cli",
                timed_out=cli_timed_out,
            ),
            "official_cli_output": {
                "path": str(cli_output.relative_to(root)),
                "sha256": _sha256_file(cli_output),
            },
        })
    dependency_lock = [
        {
            "path": relative,
            "sha256": TOOFLASHY_PARITY_SOURCE_HASHES[relative],
        }
        for relative in ("pyproject.toml", "uv.lock")
    ]
    receipt = {
        "schema": TOOFLASHY_PARITY_ADAPTER_SCHEMA,
        "adapter_source_sha256": adapter_hash,
        "upstream": upstream,
        "dependency_lock": dependency_lock,
        "census_receipt": {
            "path": str(census_path),
            "sha256": _sha256_file(census_path),
            "artifact_root": str(Path(census_artifact_root).resolve()),
        },
        "canonical_input": {
            "video": canonical_contract["canonical_video"],
            "conversion_receipt": canonical_contract["conversion_receipt"],
            "frame_map_sha256": canonical_contract["frame_map_sha256"],
        },
        "conformance_manifest": frozen_fixtures,
        "environment_contract": {
            "environment": environment,
            "sha256": _canonical_json_sha256(environment),
        },
        "runner": {"uv": str(uv), "uv_sha256": _sha256_file(uv), "timeout_seconds": timeout_seconds},
        "fixtures": fixture_rows,
        "status": "NOT_VERIFIED",
        "parity_reason": "verification_pending",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "scoreable_blockers": [
            "detector_scoring_out_of_scope",
            "independent_execution_witness_missing",
            "independent_gold_not_verified",
        ],
    }
    receipt_path = root / "tooflashy-parity-adapter-receipt.json"
    _write_json(receipt_path, receipt)
    verified = verify_tooflashy_parity_adapter(receipt_path, video, conversion)
    _write_json(receipt_path, verified)
    return {**verified, "receipt": str(receipt_path)}


def _load_tooflashy_adapter_artifact(root: Path, reference: object, *, label: str) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ExternalLeagueError(f"{label} path is not run-owned")
    path = (root / reference).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExternalLeagueError(f"{label} path escapes adapter root") from exc
    if not path.is_file():
        raise ExternalLeagueError(f"{label} artifact is missing")
    return path


def _tooflashy_child_runtime_is_frozen(
    runtime: object,
    *,
    parent_environment: Mapping[str, str],
    checkout: Path,
    root: Path,
    canonical_ffmpeg_sha256: str,
) -> bool:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python_executable", "python_executable_sha256", "python_version",
        "environment", "sys_path", "dependency_versions", "dependency_evidence",
        "decoder_executables",
    }:
        return False
    python_value = runtime.get("python_executable")
    environment = runtime.get("environment")
    sys_path = runtime.get("sys_path")
    dependencies = runtime.get("dependency_versions")
    dependency_evidence = runtime.get("dependency_evidence")
    decoder_executables = runtime.get("decoder_executables")
    if (
        not isinstance(python_value, str)
        or not isinstance(environment, Mapping)
        or not isinstance(sys_path, list)
        or not all(isinstance(item, str) for item in sys_path)
        or not isinstance(dependencies, Mapping)
        or not isinstance(dependency_evidence, Mapping)
        or not isinstance(decoder_executables, Mapping)
    ):
        return False
    python = Path(python_value)
    if not python.is_file() or runtime.get("python_executable_sha256") != _sha256_file(python):
        return False
    environment_root = root / "uv-environment"
    expected_keys = {
        "HOME", "LC_CTYPE", "LD_LIBRARY_PATH", "PATH", "PYTHONHASHSEED",
        "PYTHONNOUSERSITE", "QT_QPA_FONTDIR", "QT_QPA_PLATFORM_PLUGIN_PATH",
        "UV", "UV_CACHE_DIR", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT",
        "UV_RUN_RECURSION_DEPTH", "VIRTUAL_ENV",
    }
    if set(environment) != expected_keys:
        return False
    if any(
        environment.get(key) != value
        for key, value in parent_environment.items()
        if key != "PATH"
    ):
        return False
    if (
        environment.get("PATH") != f"{environment_root}/bin:/usr/bin:/bin"
        or environment.get("LC_CTYPE") != "C.UTF-8"
        or environment.get("UV") != str(_uv_executable())
        or environment.get("UV_RUN_RECURSION_DEPTH") != "1"
        or environment.get("VIRTUAL_ENV") != str(environment_root)
    ):
        return False
    for key in ("QT_QPA_FONTDIR", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        try:
            Path(str(environment[key])).resolve().relative_to(environment_root.resolve())
        except (KeyError, ValueError):
            return False
    library_paths = [item for item in str(environment.get("LD_LIBRARY_PATH", "")).split(":") if item]
    if not library_paths:
        return False
    try:
        for path in library_paths:
            Path(path).resolve().relative_to(environment_root.resolve())
    except ValueError:
        return False
    if dependencies != {
        "tooflashy": "0.1.0",
        "numpy": "2.4.4",
        "opencv-python": "4.13.0.92",
        "opencv-python-headless": None,
    }:
        return False
    if set(dependency_evidence) != set(dependencies):
        return False
    for name, version in dependencies.items():
        evidence = dependency_evidence.get(name)
        if version is None:
            if evidence is not None:
                return False
            continue
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != {"version", "file_count", "files_sha256"}
            or evidence.get("version") != version
            or isinstance(evidence.get("file_count"), bool)
            or not isinstance(evidence.get("file_count"), int)
            or int(evidence["file_count"]) <= 0
            or not isinstance(evidence.get("files_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", evidence["files_sha256"]) is None
        ):
            return False
    if set(decoder_executables) != {"ffmpeg", "ffprobe"}:
        return False
    for name in ("ffmpeg", "ffprobe"):
        value = shutil.which(name, path="/usr/bin:/bin")
        if value is None:
            return False
        executable = Path(value).resolve()
        version = subprocess.run([str(executable), "-version"], capture_output=True, check=False)
        expected = {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "version_stdout_sha256": _sha256_bytes(version.stdout),
            "version_stderr_sha256": _sha256_bytes(version.stderr),
        }
        if version.returncode != 0 or decoder_executables.get(name) != expected:
            return False
    if decoder_executables["ffmpeg"]["sha256"] != canonical_ffmpeg_sha256:
        return False
    allowed_roots = (
        checkout.resolve(),
        environment_root.resolve(),
        Path("/usr/lib").resolve(),
    )
    for item in sys_path:
        if item == "":
            continue
        resolved = Path(item).resolve()
        if not any(resolved == allowed or allowed in resolved.parents for allowed in allowed_roots):
            return False
    return isinstance(runtime.get("python_version"), str) and bool(runtime["python_version"])


def _fresh_replay_tooflashy_fixture(
    fixture: Path,
    *,
    checkout: Path,
    uv: Path,
    adapter_source_sha256: str,
    timeout_seconds: int,
    replay_parent: Path,
    canonical_ffmpeg_sha256: str,
) -> dict[str, object]:
    """Re-execute both paths so a caller-authored receipt cannot self-attest."""
    with tempfile.TemporaryDirectory(prefix="tooflashy-replay-", dir=replay_parent) as temporary:
        root = Path(temporary).resolve()
        (root / "home").mkdir()
        environment = {
            "HOME": str(root / "home"),
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "UV_PROJECT": str(checkout),
            "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
        }
        adapter_output = root / "adapter-output.json"
        adapter_command = [
            str(uv), "run", "--locked", "--project", str(checkout),
            "--directory", str(checkout), "python", "-c",
            _TOOFLASHY_PARITY_ADAPTER_SCRIPT, str(fixture), str(adapter_output),
            adapter_source_sha256,
        ]
        cli_command = [
            str(uv), "run", "--locked", "--project", str(checkout),
            "--directory", str(checkout), "tooflashy", "--json", str(fixture),
        ]
        try:
            adapter = subprocess.run(
                adapter_command,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=checkout,
                env=environment,
            )
            official = subprocess.run(
                cli_command,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=checkout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalLeagueError("TooFlashy fresh conformance replay timed out") from exc
        if adapter.returncode != 0 or official.returncode not in {0, 1} or not adapter_output.is_file():
            raise ExternalLeagueError("TooFlashy fresh conformance replay process failed")
        try:
            adapter_payload = json.loads(adapter_output.read_text(encoding="utf-8"))
            official_payload = json.loads(official.stdout.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError("TooFlashy fresh conformance replay output is invalid") from exc
        runtime_frozen = (
            isinstance(adapter_payload, Mapping)
            and _tooflashy_child_runtime_is_frozen(
                adapter_payload.get("runtime"),
                parent_environment=environment,
                checkout=checkout,
                root=root,
                canonical_ffmpeg_sha256=canonical_ffmpeg_sha256,
            )
        )
        return {
            "adapter": adapter_payload,
            "official_cli": official_payload,
            "official_cli_exit_code": official.returncode,
            "root": str(root),
            "environment": environment,
            "runtime_frozen": runtime_frozen,
        }


def verify_tooflashy_parity_adapter(
    receipt: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
) -> dict[str, object]:
    """Reopen TooFlashy adapter, raw ledger, CLI and source evidence fail closed."""
    receipt_path = Path(receipt).resolve()
    root = receipt_path.parent
    try:
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("TooFlashy adapter receipt is unreadable") from exc
    required = {
        "schema", "adapter_source_sha256", "upstream", "dependency_lock",
        "census_receipt", "canonical_input", "conformance_manifest", "environment_contract", "runner",
        "fixtures", "status", "parity_reason", "claim_status", "scoreable",
        "scoreable_blockers",
    }
    if not isinstance(stored, Mapping) or set(stored) != required or stored.get("schema") != TOOFLASHY_PARITY_ADAPTER_SCHEMA:
        raise ExternalLeagueError("TooFlashy adapter receipt schema or fields are invalid")
    failures: list[str] = []

    def check(condition: bool, reason: str) -> None:
        if not condition:
            failures.append(reason)

    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    try:
        conversion_payload, canonical = _canonical_decoder_timeline_contract(video, conversion)
    except (ExternalLeagueError, OSError, ValueError) as exc:
        failures.append(f"canonical_contract_invalid:{exc}")
        conversion_payload = {}
        canonical = {"canonical_video": {}, "conversion_receipt": {}, "frame_map": [], "frame_map_sha256": None}
    materializer = conversion_payload.get("materializer") if isinstance(conversion_payload, Mapping) else None
    canonical_ffmpeg_sha256 = materializer.get("binary_sha256") if isinstance(materializer, Mapping) else None
    if not isinstance(canonical_ffmpeg_sha256, str):
        failures.append("canonical_ffmpeg_provenance_missing")
        canonical_ffmpeg_sha256 = ""
    adapter_hash = _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
    check(stored.get("adapter_source_sha256") == adapter_hash, "adapter_source_drift")
    census_ref = stored.get("census_receipt")
    entry: dict[str, object] | None = None
    upstream: dict[str, object] | None = None
    if not isinstance(census_ref, Mapping):
        failures.append("census_binding_missing")
    else:
        try:
            census_path = Path(str(census_ref.get("path", ""))).resolve()
            artifact_root = Path(str(census_ref.get("artifact_root", ""))).resolve()
            if census_ref.get("sha256") != _sha256_file(census_path):
                raise ExternalLeagueError("census receipt hash mismatch")
            entry, _ = _load_execution_census_entry(census_path, artifact_root, "TooFlashy")
            upstream = _verify_tooflashy_adapter_checkout(entry)
        except (ExternalLeagueError, OSError, ValueError) as exc:
            failures.append(f"upstream_provenance_invalid:{exc}")
    check(upstream is not None and stored.get("upstream") == upstream, "upstream_receipt_drift")
    expected_lock = [
        {"path": relative, "sha256": TOOFLASHY_PARITY_SOURCE_HASHES[relative]}
        for relative in ("pyproject.toml", "uv.lock")
    ]
    check(stored.get("dependency_lock") == expected_lock, "dependency_lock_drift")
    canonical_input = stored.get("canonical_input")
    check(
        isinstance(canonical_input, Mapping)
        and canonical_input.get("video") == canonical.get("canonical_video")
        and canonical_input.get("conversion_receipt") == canonical.get("conversion_receipt")
        and canonical_input.get("frame_map_sha256") == canonical.get("frame_map_sha256"),
        "canonical_input_binding_drift",
    )
    conformance_ref = stored.get("conformance_manifest")
    frozen_fixtures: dict[str, object] | None = None
    if not isinstance(conformance_ref, Mapping) or not isinstance(conformance_ref.get("path"), str):
        failures.append("conformance_manifest_binding_missing")
    else:
        try:
            frozen_fixtures = _load_tooflashy_conformance_manifest(conformance_ref["path"])
            if dict(conformance_ref) != frozen_fixtures:
                raise ExternalLeagueError("stored conformance manifest binding drifted")
        except (ExternalLeagueError, OSError, ValueError) as exc:
            failures.append(f"conformance_manifest_invalid:{exc}")
    environment_contract = stored.get("environment_contract")
    environment = environment_contract.get("environment") if isinstance(environment_contract, Mapping) else None
    expected_environment = {
        "HOME": str(root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_PROJECT": str(Path(str(upstream.get("path", ""))) if upstream else ""),
        "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
    }
    check(
        isinstance(environment, Mapping)
        and dict(environment) == expected_environment
        and environment_contract.get("sha256") == _canonical_json_sha256(expected_environment),
        "execution_environment_drift",
    )
    runner = stored.get("runner")
    uv = _uv_executable()
    check(
        isinstance(runner, Mapping)
        and runner.get("uv") == str(uv)
        and uv.is_file()
        and runner.get("uv_sha256") == _sha256_file(uv)
        and entry is not None
        and runner.get("uv_sha256") == entry.get("binary_sha256")
        and isinstance(runner.get("timeout_seconds"), int)
        and not isinstance(runner.get("timeout_seconds"), bool)
        and int(runner.get("timeout_seconds", 0)) > 0,
        "runner_provenance_drift",
    )
    fixtures = stored.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) < 2:
        failures.append("conformance_fixture_population_invalid")
        fixtures = []
    observed_paths: list[Path] = []
    for expected_ordinal, fixture_row in enumerate(fixtures):
        prefix = f"fixture_{expected_ordinal}"
        if not isinstance(fixture_row, Mapping):
            failures.append(f"{prefix}:row_invalid")
            continue
        allowed_row_fields = {
            "ordinal", "fixture_id", "role", "input", "adapter_process", "adapter_output",
            "official_cli_process", "official_cli_output",
        }
        check(set(fixture_row) == allowed_row_fields, f"{prefix}:unsupported_receipt_field")
        check(fixture_row.get("ordinal") == expected_ordinal, f"{prefix}:ordinal_drift")
        expected_fixture_id = (
            "canonical"
            if expected_ordinal == 0
            else (
                frozen_fixtures["fixtures"][expected_ordinal - 1]["id"]
                if frozen_fixtures is not None and expected_ordinal - 1 < len(frozen_fixtures["fixtures"])
                else None
            )
        )
        check(fixture_row.get("fixture_id") == expected_fixture_id, f"{prefix}:fixture_identity_drift")
        check(
            fixture_row.get("role") == ("CANONICAL" if expected_ordinal == 0 else "CONFORMANCE"),
            f"{prefix}:role_drift",
        )
        input_ref = fixture_row.get("input")
        try:
            if not isinstance(input_ref, Mapping) or set(input_ref) != {"path", "sha256", "bytes"}:
                raise ExternalLeagueError("input reference fields invalid")
            fixture_path = Path(str(input_ref["path"])).resolve()
            if (
                not fixture_path.is_file()
                or input_ref.get("sha256") != _sha256_file(fixture_path)
                or input_ref.get("bytes") != fixture_path.stat().st_size
            ):
                raise ExternalLeagueError("input hash or size mismatch")
            observed_paths.append(fixture_path)
        except (ExternalLeagueError, OSError, KeyError) as exc:
            failures.append(f"{prefix}:input_invalid:{exc}")
            continue
        adapter_process = fixture_row.get("adapter_process")
        cli_process = fixture_row.get("official_cli_process")
        adapter_ref = fixture_row.get("adapter_output")
        cli_ref = fixture_row.get("official_cli_output")
        if not all(isinstance(item, Mapping) for item in (adapter_process, cli_process, adapter_ref, cli_ref)):
            failures.append(f"{prefix}:process_or_output_reference_invalid")
            continue
        try:
            adapter_output = _load_tooflashy_adapter_artifact(root, adapter_ref.get("path"), label="adapter output")
            cli_output = _load_tooflashy_adapter_artifact(root, cli_ref.get("path"), label="official CLI output")
            if adapter_ref.get("exists") is not True or adapter_ref.get("sha256") != _sha256_file(adapter_output):
                raise ExternalLeagueError("adapter output hash mismatch")
            if cli_ref.get("sha256") != _sha256_file(cli_output):
                raise ExternalLeagueError("official CLI output hash mismatch")
            adapter_payload = json.loads(adapter_output.read_text(encoding="utf-8"))
            cli_payload = json.loads(cli_output.read_text(encoding="utf-8"))
        except (ExternalLeagueError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"{prefix}:raw_artifact_invalid:{exc}")
            continue
        checkout = Path(str(upstream.get("path", ""))) if upstream else Path()
        expected_adapter_command = [
            str(uv), "run", "--locked", "--project", str(checkout), "--directory", str(checkout),
            "python", "-c", _TOOFLASHY_PARITY_ADAPTER_SCRIPT, str(fixture_path), str(adapter_output), adapter_hash,
        ]
        expected_cli_command = [
            str(uv), "run", "--locked", "--project", str(checkout), "--directory", str(checkout),
            "tooflashy", "--json", str(fixture_path),
        ]
        for process_name, process, command, allowed_exit_codes in (
            ("adapter", adapter_process, expected_adapter_command, {0}),
            ("official_cli", cli_process, expected_cli_command, {0, 1}),
        ):
            required_process_fields = {
                "command", "working_directory", "environment_sha256", "exit_code",
                "timed_out", "stdout", "stderr",
            }
            check(set(process) == required_process_fields, f"{prefix}:{process_name}_process_fields_invalid")
            check(process.get("command") == command, f"{prefix}:{process_name}_command_drift")
            check(process.get("working_directory") == str(checkout), f"{prefix}:{process_name}_cwd_drift")
            check(process.get("environment_sha256") == _canonical_json_sha256(expected_environment), f"{prefix}:{process_name}_environment_drift")
            check(process.get("timed_out") is False, f"{prefix}:{process_name}_timeout")
            check(
                not isinstance(process.get("exit_code"), bool)
                and isinstance(process.get("exit_code"), int)
                and process.get("exit_code") in allowed_exit_codes,
                f"{prefix}:{process_name}_exit_invalid",
            )
            for stream_name in ("stdout", "stderr"):
                stream_ref = process.get(stream_name)
                try:
                    if not isinstance(stream_ref, Mapping) or set(stream_ref) != {"path", "sha256"}:
                        raise ExternalLeagueError("stream reference invalid")
                    stream_path = (root / f"fixture-{expected_ordinal:03d}" / str(stream_ref["path"])).resolve()
                    stream_path.relative_to(root / f"fixture-{expected_ordinal:03d}")
                    if not stream_path.is_file() or stream_ref.get("sha256") != _sha256_file(stream_path):
                        raise ExternalLeagueError("stream hash mismatch")
                except (ExternalLeagueError, OSError, ValueError, KeyError) as exc:
                    failures.append(f"{prefix}:{process_name}_{stream_name}_invalid:{exc}")
        try:
            cli_stdout_path = (
                root
                / f"fixture-{expected_ordinal:03d}"
                / str(cli_process["stdout"]["path"])
            ).resolve()
            check(
                cli_stdout_path.read_bytes() == cli_output.read_bytes(),
                f"{prefix}:official_cli_stdout_output_divergence",
            )
        except (OSError, KeyError, TypeError):
            failures.append(f"{prefix}:official_cli_stdout_unreadable")
        if not isinstance(adapter_payload, Mapping) or set(adapter_payload) != {
            "schema", "evidence_origin", "adapter_source_sha256", "input", "public_api",
            "runtime", "decode", "result",
        }:
            failures.append(f"{prefix}:adapter_payload_fields_invalid")
            continue
        check(adapter_payload.get("schema") == "flashpatch-l7-tooflashy-child-adapter-v1", f"{prefix}:adapter_schema_invalid")
        check(adapter_payload.get("evidence_origin") == "live_generator_append_immediately_before_yield_v1", f"{prefix}:synthetic_post_hoc_audit")
        check(adapter_payload.get("adapter_source_sha256") == adapter_hash, f"{prefix}:adapter_source_unbound")
        check(adapter_payload.get("input") == {"path": str(fixture_path), "sha256": _sha256_file(fixture_path)}, f"{prefix}:adapter_input_drift")
        public_api = adapter_payload.get("public_api")
        if not isinstance(public_api, Mapping) or set(public_api) != set(TOOFLASHY_PARITY_CALLABLE_HASHES):
            failures.append(f"{prefix}:public_api_evidence_invalid")
        else:
            expected_modules = {
                "iter_video_frames": ("tooflashy.video", "iter_video_frames", "src/tooflashy/video.py"),
                "analyze_frames": ("tooflashy.analysis", "analyze_frames", "src/tooflashy/analysis.py"),
            }
            for api_name, (module_name, qualname, relative) in expected_modules.items():
                api = public_api.get(api_name)
                check(
                    isinstance(api, Mapping)
                    and set(api) == {"module", "qualname", "module_path", "module_sha256", "callable_source_sha256"}
                    and api.get("module") == module_name
                    and api.get("qualname") == qualname
                    and Path(str(api.get("module_path", ""))).resolve() == (checkout / relative).resolve()
                    and api.get("module_sha256") == TOOFLASHY_PARITY_SOURCE_HASHES[relative]
                    and api.get("callable_source_sha256") == TOOFLASHY_PARITY_CALLABLE_HASHES[api_name],
                    f"{prefix}:public_api_altered:{api_name}",
                )
        runtime = adapter_payload.get("runtime")
        check(
            _tooflashy_child_runtime_is_frozen(
                runtime,
                parent_environment=expected_environment,
                checkout=checkout,
                root=root,
                canonical_ffmpeg_sha256=canonical_ffmpeg_sha256,
            ),
            f"{prefix}:runtime_provenance_invalid",
        )
        decode = adapter_payload.get("decode")
        if not isinstance(decode, Mapping) or set(decode) != {"engine", "fps", "frame_count", "pixel_format", "ledger", "ledger_sha256"}:
            failures.append(f"{prefix}:decode_ledger_fields_invalid")
            continue
        ledger = decode.get("ledger")
        fps = decode.get("fps")
        if (
            decode.get("engine") != "ffmpeg"
            or decode.get("pixel_format") != "rgb24"
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or float(fps) <= 0
            or not isinstance(ledger, list)
            or decode.get("frame_count") != len(ledger)
            or decode.get("ledger_sha256") != _canonical_json_sha256(ledger)
        ):
            failures.append(f"{prefix}:decode_ledger_invalid")
            continue
        for index, frame in enumerate(ledger):
            check(
                isinstance(frame, Mapping)
                and set(frame) == {"index", "cfr_timestamp_us", "shape", "pixel_format", "rgb_sha256"}
                and frame.get("index") == index
                and frame.get("cfr_timestamp_us") == round(index * 1_000_000 / float(fps))
                and isinstance(frame.get("shape"), list)
                and len(frame["shape"]) == 3
                and frame["shape"][-1] == 3
                and frame.get("pixel_format") == "rgb24"
                and isinstance(frame.get("rgb_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", frame["rgb_sha256"]) is not None,
                f"{prefix}:frame_ledger_drift:{index}",
            )
        if expected_ordinal == 0:
            canonical_rows = [
                {
                    "index": row["frame_index"],
                    "cfr_timestamp_us": row["cfr_timestamp_us"],
                    "shape": list(canonical["shape"][1:]),
                    "pixel_format": "rgb24",
                    "rgb_sha256": row["rgb_sha256"],
                }
                for row in canonical.get("frame_map", [])
            ]
            check(ledger == canonical_rows, f"{prefix}:canonical_rgb_timeline_drift")
        if not isinstance(cli_payload, Mapping) or set(cli_payload) != TOOFLASHY_OFFICIAL_JSON_FIELDS:
            failures.append(f"{prefix}:official_cli_unsupported_output_field")
            continue
        check(
            isinstance(cli_payload.get("path"), str)
            and isinstance(cli_payload.get("passes"), bool)
            and not isinstance(cli_payload.get("fps"), bool)
            and isinstance(cli_payload.get("fps"), (int, float))
            and float(cli_payload["fps"]) > 0
            and not isinstance(cli_payload.get("frame_count"), bool)
            and isinstance(cli_payload.get("frame_count"), int)
            and int(cli_payload["frame_count"]) >= 0
            and not isinstance(cli_payload.get("event_count"), bool)
            and isinstance(cli_payload.get("event_count"), int)
            and int(cli_payload["event_count"]) >= 0
            and isinstance(cli_payload.get("failures"), list)
            and all(isinstance(item, str) for item in cli_payload["failures"]),
            f"{prefix}:official_cli_value_types_invalid",
        )
        result = adapter_payload.get("result")
        expected_result_fields = {*TOOFLASHY_OFFICIAL_JSON_FIELDS, "event_representation"}
        if not isinstance(result, Mapping) or set(result) != expected_result_fields:
            failures.append(f"{prefix}:adapter_result_fields_invalid")
            continue
        official_projection = {field: result.get(field) for field in TOOFLASHY_OFFICIAL_JSON_FIELDS}
        check(official_projection == dict(cli_payload), f"{prefix}:adapter_cli_divergence")
        check(
            result.get("event_representation") == {
                "event_count": cli_payload.get("event_count"),
                "failures": cli_payload.get("failures"),
            },
            f"{prefix}:event_representation_divergence",
        )
        check(result.get("path") == str(fixture_path), f"{prefix}:result_path_drift")
        check(result.get("fps") == fps, f"{prefix}:result_fps_drift")
        check(result.get("frame_count") == len(ledger), f"{prefix}:result_frame_count_drift")
        check(isinstance(result.get("passes"), bool), f"{prefix}:result_outcome_invalid")
        check(
            cli_process.get("exit_code") == (0 if result.get("passes") is True else 1),
            f"{prefix}:official_cli_exit_outcome_divergence",
        )
        try:
            fresh = _fresh_replay_tooflashy_fixture(
                fixture_path,
                checkout=checkout,
                uv=uv,
                adapter_source_sha256=adapter_hash,
                timeout_seconds=int(runner["timeout_seconds"]),
                replay_parent=root,
                canonical_ffmpeg_sha256=canonical_ffmpeg_sha256,
            )
            fresh_adapter = fresh.get("adapter")
            fresh_cli = fresh.get("official_cli")
            if not isinstance(fresh_adapter, Mapping) or not isinstance(fresh_cli, Mapping):
                raise ExternalLeagueError("fresh replay payload root is invalid")
            check(fresh.get("runtime_frozen") is True, f"{prefix}:fresh_replay_runtime_invalid")
            for field in (
                "schema", "evidence_origin", "adapter_source_sha256", "input",
                "public_api", "decode", "result",
            ):
                check(
                    fresh_adapter.get(field) == adapter_payload.get(field),
                    f"{prefix}:fresh_replay_adapter_divergence:{field}",
                )
            fresh_runtime = fresh_adapter.get("runtime")
            if not isinstance(fresh_runtime, Mapping) or not isinstance(runtime, Mapping):
                failures.append(f"{prefix}:fresh_replay_runtime_evidence_missing")
            else:
                for field in (
                    "python_executable", "python_executable_sha256", "python_version",
                    "dependency_versions", "dependency_evidence", "decoder_executables",
                ):
                    check(
                        fresh_runtime.get(field) == runtime.get(field),
                        f"{prefix}:fresh_replay_runtime_divergence:{field}",
                    )
            check(dict(fresh_cli) == dict(cli_payload), f"{prefix}:fresh_replay_official_cli_divergence")
            check(
                fresh.get("official_cli_exit_code") == cli_process.get("exit_code"),
                f"{prefix}:fresh_replay_exit_divergence",
            )
        except (ExternalLeagueError, OSError, ValueError, KeyError) as exc:
            failures.append(f"{prefix}:fresh_replay_failed:{exc}")
    check(bool(observed_paths) and observed_paths[0] == video, "canonical_fixture_path_drift")
    check(len(observed_paths) == len(set(observed_paths)), "duplicate_conformance_fixture")
    if len(observed_paths) == len(fixtures):
        check(
            len({_sha256_file(path) for path in observed_paths}) == len(observed_paths),
            "duplicate_conformance_fixture_content",
        )
    result = dict(stored)
    result.update({
        "status": "VERIFIED" if not failures else "NOT_VERIFIED",
        "parity_reason": None if not failures else failures,
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "scoreable_blockers": [
            "detector_scoring_out_of_scope",
            "independent_execution_witness_missing",
            "independent_gold_not_verified",
        ],
    })
    return result


def _tooflashy_private_storage_roots(
    receipt_path: Path,
    verified: Mapping[str, object],
) -> dict[str, Path]:
    upstream = verified.get("upstream")
    environment_contract = verified.get("environment_contract")
    environment = (
        environment_contract.get("environment")
        if isinstance(environment_contract, Mapping)
        else None
    )
    if not isinstance(upstream, Mapping) or not isinstance(environment, Mapping):
        raise ExternalLeagueError("TooFlashy copied replay storage binding is missing")
    expected = {
        "source_checkout": Path(str(upstream.get("path", ""))).resolve(),
        "uv_environment": (receipt_path.parent / "uv-environment").resolve(),
        "uv_cache": (receipt_path.parent / "uv-cache").resolve(),
    }
    if (
        environment.get("UV_PROJECT") != str(expected["source_checkout"])
        or environment.get("UV_PROJECT_ENVIRONMENT") != str(expected["uv_environment"])
        or environment.get("UV_CACHE_DIR") != str(expected["uv_cache"])
    ):
        raise ExternalLeagueError("TooFlashy copied replay private roots drifted from the adapter receipt")
    roots = list(expected.values())
    if len(set(roots)) != len(roots):
        raise ExternalLeagueError("TooFlashy copied replay private roots are not distinct")
    for label, root in expected.items():
        if not root.is_dir() or root.is_symlink():
            raise ExternalLeagueError(f"TooFlashy copied replay private root is unavailable: {label}")
    return expected


def _tooflashy_storage_entries(root: Path, *, source_checkout: bool) -> list[Path]:
    if source_checkout:
        entries = [(root / relative) for relative in TOOFLASHY_PARITY_SOURCE_HASHES]
        if any(not entry.is_file() or entry.is_symlink() for entry in entries):
            raise ExternalLeagueError("TooFlashy copied replay source closure is incomplete")
        return sorted(entries, key=lambda path: str(path.relative_to(root)))

    entries: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                entries.append(candidate)
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        entries.extend(directory_path / name for name in sorted(file_names))
    if not entries:
        raise ExternalLeagueError("TooFlashy copied replay private storage closure is empty")
    return sorted(entries, key=lambda path: str(path.relative_to(root)))


def _tooflashy_private_storage_manifest(
    roots: Mapping[str, Path],
) -> tuple[dict[str, object], set[tuple[int, int]]]:
    manifests: dict[str, object] = {}
    identities: set[tuple[int, int]] = set()
    for label in ("source_checkout", "uv_environment", "uv_cache"):
        root = roots[label]
        rows: list[dict[str, object]] = []
        root_identities: set[tuple[int, int]] = set()
        for path in _tooflashy_storage_entries(root, source_checkout=label == "source_checkout"):
            try:
                observed = path.lstat()
            except OSError as exc:
                raise ExternalLeagueError(
                    f"TooFlashy copied replay storage entry cannot be inspected: {label}"
                ) from exc
            identity = (int(observed.st_dev), int(observed.st_ino))
            root_identities.add(identity)
            identities.add(identity)
            relative = str(path.relative_to(root))
            if stat.S_ISREG(observed.st_mode):
                rows.append({
                    "path": relative,
                    "type": "file",
                    "bytes": observed.st_size,
                    "sha256": _sha256_file(path),
                })
            elif stat.S_ISLNK(observed.st_mode):
                rows.append({
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                })
            else:
                raise ExternalLeagueError(
                    f"TooFlashy copied replay private storage contains a non-file artifact: {label}"
                )
        manifests[label] = {
            "root": str(root),
            "entry_count": len(rows),
            "storage_identity_count": len(root_identities),
            "tree_sha256": _canonical_json_sha256(rows),
        }
    return manifests, identities


def _tooflashy_regular_file_bytes(path: Path, root: Path, *, label: str) -> bytes:
    try:
        root_resolved = root.resolve(strict=True)
        lexical = Path(os.path.abspath(path))
        lexical_relative = lexical.relative_to(root_resolved)
        current = root_resolved
        for part in lexical_relative.parts:
            current = current / part
            if current.is_symlink():
                raise ExternalLeagueError(f"{label} traverses a symlink")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ExternalLeagueError(f"{label} escapes its expected root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ExternalLeagueError(f"{label} is not a regular file")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise ExternalLeagueError(f"{label} is unreadable") from exc


def _tooflashy_copied_source_projection(
    checkout: Path,
    upstream: Mapping[str, object],
) -> dict[str, object]:
    frozen_hashes = upstream.get("source_sha256")
    if (
        upstream.get("repository_url") != "https://github.com/hashb/TooFlashy"
        or upstream.get("revision") != TOOFLASHY_PARITY_ADAPTER_REVISION
        or upstream.get("tree") != TOOFLASHY_PARITY_ADAPTER_TREE
        or upstream.get("license") != "Apache-2.0"
        or not isinstance(frozen_hashes, Mapping)
        or set(frozen_hashes) != set(TOOFLASHY_PARITY_SOURCE_HASHES)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in frozen_hashes.values()
        )
    ):
        raise ExternalLeagueError("TooFlashy copied replay source provenance is not frozen")
    observed: dict[str, str] = {}
    for relative in TOOFLASHY_PARITY_SOURCE_HASHES:
        expected = str(frozen_hashes[relative])
        payload = _tooflashy_regular_file_bytes(
            checkout / relative,
            checkout,
            label=f"TooFlashy copied replay source file {relative}",
        )
        observed[relative] = _sha256_bytes(payload)
        if observed[relative] != expected:
            raise ExternalLeagueError(f"TooFlashy copied replay source drifted: {relative}")
    source_root = checkout / "src"
    if not source_root.is_dir() or source_root.is_symlink():
        raise ExternalLeagueError("TooFlashy copied replay editable source root is unavailable")
    return {
        "repository_url": upstream["repository_url"],
        "revision": upstream["revision"],
        "tree": upstream["tree"],
        "license": upstream["license"],
        "source_sha256": observed,
    }


def _tooflashy_record_digest(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii").rstrip("=")


def _tooflashy_uv_timestamp(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "secs_since_epoch", "nanos_since_epoch",
    }:
        raise ExternalLeagueError(f"TooFlashy editable {label} is invalid")
    seconds = value.get("secs_since_epoch")
    nanoseconds = value.get("nanos_since_epoch")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or seconds < 0
        or isinstance(nanoseconds, bool)
        or not isinstance(nanoseconds, int)
        or not 0 <= nanoseconds < 1_000_000_000
    ):
        raise ExternalLeagueError(f"TooFlashy editable {label} is invalid")
    return {"secs_since_epoch": seconds, "nanos_since_epoch": nanoseconds}


def _tooflashy_editable_installation_projection(
    environment_root: Path,
    checkout: Path,
    raw_evidence: object,
) -> dict[str, object]:
    """Reopen one editable install and erase only proven generator-local values."""
    if (
        not isinstance(raw_evidence, Mapping)
        or set(raw_evidence) != {"version", "file_count", "files_sha256"}
        or raw_evidence.get("version") != "0.1.0"
    ):
        raise ExternalLeagueError("TooFlashy editable dependency evidence is invalid")
    candidates = sorted(
        environment_root.glob("lib/python*/site-packages/tooflashy-0.1.0.dist-info")
    )
    if len(candidates) != 1:
        raise ExternalLeagueError("TooFlashy editable dist-info identity is ambiguous")
    dist_info = candidates[0]
    site_packages = dist_info.parent
    if dist_info.is_symlink() or not dist_info.is_dir():
        raise ExternalLeagueError("TooFlashy editable dist-info is not a regular directory")
    try:
        dist_info.resolve(strict=True).relative_to(environment_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ExternalLeagueError("TooFlashy editable dist-info escapes its environment") from exc

    expected_dist_files = {
        *TOOFLASHY_EDITABLE_SEMANTIC_FILES,
        "RECORD",
        "direct_url.json",
        "uv_cache.json",
    }
    actual_dist_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(dist_info, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ExternalLeagueError("TooFlashy editable dist-info contains a symlink")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                raise ExternalLeagueError("TooFlashy editable dist-info contains a symlink")
            actual_dist_files.add(str(path.relative_to(dist_info)))
    if actual_dist_files != expected_dist_files:
        raise ExternalLeagueError("TooFlashy editable dist-info contains unallowlisted drift")

    semantic_rows: dict[str, dict[str, object]] = {}
    for name, expected in TOOFLASHY_EDITABLE_SEMANTIC_FILES.items():
        payload = _tooflashy_regular_file_bytes(
            dist_info / name,
            environment_root,
            label=f"TooFlashy editable semantic file {name}",
        )
        if payload != expected:
            raise ExternalLeagueError(f"TooFlashy editable semantic file drifted: {name}")
        semantic_rows[name] = {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    direct_payload = _tooflashy_regular_file_bytes(
        dist_info / "direct_url.json",
        environment_root,
        label="TooFlashy editable direct_url.json",
    )
    try:
        direct = json.loads(direct_payload)
    except json.JSONDecodeError as exc:
        raise ExternalLeagueError("TooFlashy editable direct_url.json is invalid") from exc
    direct_url = direct.get("url") if isinstance(direct, Mapping) else None
    parts = urllib.parse.urlsplit(direct_url) if isinstance(direct_url, str) else None
    if (
        not isinstance(direct, Mapping)
        or set(direct) != {"url", "dir_info"}
        or direct.get("dir_info") != {"editable": True}
        or parts is None
        or parts.scheme != "file"
        or parts.netloc != ""
        or parts.query != ""
        or parts.fragment != ""
        or urllib.parse.unquote(parts.path) != str(checkout)
        or direct_url != checkout.as_uri()
    ):
        raise ExternalLeagueError("TooFlashy editable direct_url target is not its own checkout")

    pth_candidates = sorted(site_packages.glob("tooflashy*.pth"))
    if pth_candidates != [site_packages / "tooflashy.pth"]:
        raise ExternalLeagueError("TooFlashy editable PTH identity is ambiguous")
    pth_payload = _tooflashy_regular_file_bytes(
        site_packages / "tooflashy.pth",
        environment_root,
        label="TooFlashy editable PTH",
    )
    if pth_payload != str(checkout / "src").encode("utf-8"):
        raise ExternalLeagueError("TooFlashy editable PTH target is not its own checkout src")

    cache_payload = _tooflashy_regular_file_bytes(
        dist_info / "uv_cache.json",
        environment_root,
        label="TooFlashy editable uv_cache.json",
    )
    try:
        cache = json.loads(cache_payload)
    except json.JSONDecodeError as exc:
        raise ExternalLeagueError("TooFlashy editable uv_cache.json is invalid") from exc
    if (
        not isinstance(cache, Mapping)
        or set(cache) != {"timestamp", "commit", "tags", "env", "directories"}
        or cache.get("commit") is not None
        or cache.get("tags") is not None
        or cache.get("env") != {}
        or not isinstance(cache.get("directories"), Mapping)
        or set(cache["directories"]) != {"src"}
    ):
        raise ExternalLeagueError("TooFlashy editable uv_cache semantic fields drifted")
    timestamp = _tooflashy_uv_timestamp(cache["timestamp"], label="uv_cache timestamp")
    source_timestamp = _tooflashy_uv_timestamp(
        cache["directories"]["src"],
        label="uv_cache source timestamp",
    )
    if timestamp != source_timestamp:
        raise ExternalLeagueError("TooFlashy editable uv_cache timestamps disagree")

    console_candidates = sorted(
        path for path in (environment_root / "bin").glob("tooflashy*") if path.is_file()
    )
    console = environment_root / "bin" / "tooflashy"
    if console_candidates != [console]:
        raise ExternalLeagueError("TooFlashy editable console-script identity is ambiguous")
    console_payload = _tooflashy_regular_file_bytes(
        console,
        environment_root,
        label="TooFlashy editable console script",
    )
    shebang, separator, console_body = console_payload.partition(b"\n")
    if (
        separator != b"\n"
        or shebang != b"#!" + str(environment_root / "bin" / "python").encode("utf-8")
        or console_body != TOOFLASHY_EDITABLE_CONSOLE_BODY
    ):
        raise ExternalLeagueError("TooFlashy editable console script drifted")

    record_payload = _tooflashy_regular_file_bytes(
        dist_info / "RECORD",
        environment_root,
        label="TooFlashy editable RECORD",
    )
    try:
        record_rows = list(csv.reader(io.StringIO(record_payload.decode("utf-8")), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ExternalLeagueError("TooFlashy editable RECORD is invalid") from exc
    if any(len(row) != 3 for row in record_rows):
        raise ExternalLeagueError("TooFlashy editable RECORD row shape is invalid")
    by_path = {row[0]: row for row in record_rows}
    dist_prefix = dist_info.name
    console_relative = "../../../bin/tooflashy"
    pth_relative = "tooflashy.pth"
    direct_relative = f"{dist_prefix}/direct_url.json"
    cache_relative = f"{dist_prefix}/uv_cache.json"
    record_relative = f"{dist_prefix}/RECORD"
    semantic_relatives = {
        f"{dist_prefix}/{name}" for name in TOOFLASHY_EDITABLE_SEMANTIC_FILES
    }
    expected_record_paths = {
        console_relative,
        pth_relative,
        direct_relative,
        cache_relative,
        record_relative,
        *semantic_relatives,
    }
    if len(by_path) != len(record_rows) or set(by_path) != expected_record_paths:
        raise ExternalLeagueError("TooFlashy editable RECORD contains unallowlisted drift")

    normalized_record: list[dict[str, object]] = []
    path_bound = {
        console_relative: "ENVIRONMENT_SHEBANG",
        pth_relative: "OWN_CHECKOUT_SRC",
        direct_relative: "OWN_CHECKOUT_URL",
        cache_relative: "UV_BUILD_TIMESTAMP",
        record_relative: "GENERATED_RECORD",
    }
    installed_rows: list[dict[str, object]] = []
    for relative in sorted(expected_record_paths):
        row = by_path[relative]
        path = (site_packages / relative).resolve()
        try:
            path.relative_to(environment_root.resolve())
        except ValueError as exc:
            raise ExternalLeagueError("TooFlashy editable RECORD path escapes its environment") from exc
        if relative == record_relative:
            if row[1:] != ["", ""]:
                raise ExternalLeagueError("TooFlashy editable RECORD self row is invalid")
        else:
            payload = _tooflashy_regular_file_bytes(
                path,
                environment_root,
                label=f"TooFlashy editable RECORD file {relative}",
            )
            if (
                row[1] != f"sha256={_tooflashy_record_digest(payload)}"
                or row[2] != str(len(payload))
            ):
                raise ExternalLeagueError(f"TooFlashy editable RECORD digest drifted: {relative}")
            if path.name not in {"RECORD", "INSTALLER", "REQUESTED"}:
                try:
                    path.relative_to(site_packages.resolve())
                except ValueError:
                    pass
                else:
                    installed_rows.append({
                        "path": relative,
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                    })
        if relative in path_bound:
            normalized_record.append({"path": relative, "generated_binding": path_bound[relative]})
        else:
            normalized_record.append({
                "path": relative,
                "bytes": int(row[2]),
                "sha256": row[1].removeprefix("sha256="),
            })

    expected_raw = {
        "version": "0.1.0",
        "file_count": len(installed_rows),
        "files_sha256": _canonical_json_sha256(installed_rows),
    }
    if dict(raw_evidence) != expected_raw:
        raise ExternalLeagueError("TooFlashy editable dependency evidence drifted from live files")
    projection: dict[str, object] = {
        "version": "0.1.0",
        "semantic_files": semantic_rows,
        "direct_url": {"editable": True, "target": "OWN_CHECKOUT"},
        "pth": {"target": "OWN_CHECKOUT/src"},
        "uv_cache": {
            "timestamp": "UV_BUILD_TIMESTAMP",
            "commit": None,
            "tags": None,
            "env": {},
            "directories": {"src": "UV_BUILD_TIMESTAMP"},
        },
        "console_script": {
            "shebang": "OWN_ENVIRONMENT/bin/python",
            "body_sha256": _sha256_bytes(console_body),
        },
        "normalized_record": normalized_record,
        "normalized_record_sha256": _canonical_json_sha256(normalized_record),
    }
    projection["semantic_sha256"] = _canonical_json_sha256(projection)
    return projection


def _tooflashy_copied_replay_projection(
    receipt_path: Path,
    verified: Mapping[str, object],
) -> dict[str, object]:
    upstream = verified.get("upstream")
    fixtures = verified.get("fixtures")
    if not isinstance(upstream, Mapping) or not isinstance(fixtures, list) or not fixtures:
        raise ExternalLeagueError("TooFlashy copied replay adapter fixtures are invalid")
    checkout = Path(str(upstream.get("path", ""))).resolve()
    roots = _tooflashy_private_storage_roots(receipt_path, verified)
    source_projection = _tooflashy_copied_source_projection(checkout, upstream)
    projected_fixtures: list[dict[str, object]] = []
    for ordinal, fixture in enumerate(fixtures):
        if not isinstance(fixture, Mapping):
            raise ExternalLeagueError("TooFlashy copied replay fixture row is invalid")
        adapter_ref = fixture.get("adapter_output")
        cli_ref = fixture.get("official_cli_output")
        if not isinstance(adapter_ref, Mapping) or not isinstance(cli_ref, Mapping):
            raise ExternalLeagueError("TooFlashy copied replay fixture output binding is missing")
        adapter_path = _load_tooflashy_adapter_artifact(
            receipt_path.parent,
            adapter_ref.get("path"),
            label="TooFlashy copied replay adapter output",
        )
        cli_path = _load_tooflashy_adapter_artifact(
            receipt_path.parent,
            cli_ref.get("path"),
            label="TooFlashy copied replay official CLI output",
        )
        try:
            adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
            official_cli = json.loads(cli_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalLeagueError("TooFlashy copied replay raw output cannot be reopened") from exc
        public_api = adapter.get("public_api") if isinstance(adapter, Mapping) else None
        runtime = adapter.get("runtime") if isinstance(adapter, Mapping) else None
        if not isinstance(public_api, Mapping) or not isinstance(runtime, Mapping):
            raise ExternalLeagueError("TooFlashy copied replay API or runtime evidence is invalid")
        dependency_evidence = runtime.get("dependency_evidence")
        if not isinstance(dependency_evidence, Mapping):
            raise ExternalLeagueError("TooFlashy copied replay dependency evidence is invalid")
        normalized_dependencies = dict(dependency_evidence)
        normalized_dependencies["tooflashy"] = _tooflashy_editable_installation_projection(
            roots["uv_environment"],
            checkout,
            dependency_evidence.get("tooflashy"),
        )
        normalized_api: dict[str, object] = {}
        for name in sorted(TOOFLASHY_PARITY_CALLABLE_HASHES):
            evidence = public_api.get(name)
            if not isinstance(evidence, Mapping):
                raise ExternalLeagueError("TooFlashy copied replay public API evidence is missing")
            module_path = Path(str(evidence.get("module_path", ""))).resolve()
            try:
                relative_module = str(module_path.relative_to(checkout))
            except ValueError as exc:
                raise ExternalLeagueError("TooFlashy copied replay module escapes its source checkout") from exc
            normalized_api[name] = {
                "module": evidence.get("module"),
                "qualname": evidence.get("qualname"),
                "module_path": relative_module,
                "module_sha256": evidence.get("module_sha256"),
                "callable_source_sha256": evidence.get("callable_source_sha256"),
            }
        input_ref = fixture.get("input")
        projected_fixtures.append({
            "ordinal": ordinal,
            "fixture_id": fixture.get("fixture_id"),
            "role": fixture.get("role"),
            "input": {
                "sha256": input_ref.get("sha256") if isinstance(input_ref, Mapping) else None,
                "bytes": input_ref.get("bytes") if isinstance(input_ref, Mapping) else None,
            },
            "adapter_identity": {
                "schema": adapter.get("schema"),
                "evidence_origin": adapter.get("evidence_origin"),
                "adapter_source_sha256": adapter.get("adapter_source_sha256"),
            },
            "public_api": normalized_api,
            "dependency_versions": runtime.get("dependency_versions"),
            "dependency_evidence": normalized_dependencies,
            "decoder_executables": runtime.get("decoder_executables"),
            "decode": adapter.get("decode"),
            "result": adapter.get("result"),
            "official_cli": official_cli,
        })
    return {
        "source": source_projection,
        "fixtures": projected_fixtures,
        "fixtures_sha256": _canonical_json_sha256(projected_fixtures),
    }


def verify_tooflashy_copied_replay_witness(
    primary_receipt: Path | str,
    replay_receipt: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    *,
    destination: Path | str | None = None,
) -> dict[str, object]:
    """Verify two storage-independent locked TooFlashy replays without scoring.

    Both adapter receipts are freshly replayed by the existing parity verifier.
    This additional gate proves that the source checkout, uv environment and uv
    cache used by the second run do not share private-storage inodes with the
    first run.  Shared immutable host tools and canonical inputs are expected.
    """
    primary_path = Path(primary_receipt).resolve()
    replay_path = Path(replay_receipt).resolve()
    if primary_path == replay_path:
        raise ExternalLeagueError("TooFlashy copied replay receipts must be distinct")
    primary = verify_tooflashy_parity_adapter(
        primary_path,
        canonical_video,
        conversion_receipt,
    )
    replay = verify_tooflashy_parity_adapter(
        replay_path,
        canonical_video,
        conversion_receipt,
    )
    for label, verified in (("primary", primary), ("replay", replay)):
        if (
            verified.get("status") != "VERIFIED"
            or verified.get("claim_status") != "NOT_SCOREABLE"
            or verified.get("scoreable") is not False
        ):
            raise ExternalLeagueError(f"TooFlashy copied replay adapter is not verified: {label}")

    primary_upstream = primary.get("upstream")
    replay_upstream = replay.get("upstream")
    if not isinstance(primary_upstream, Mapping) or not isinstance(replay_upstream, Mapping):
        raise ExternalLeagueError("TooFlashy copied replay upstream provenance is invalid")
    primary_source = Path(str(primary_upstream.get("path", ""))).resolve()
    replay_source = Path(str(replay_upstream.get("path", ""))).resolve()
    if primary_source == replay_source:
        raise ExternalLeagueError("TooFlashy copied replay must use a second source checkout")
    if (
        {key: value for key, value in primary_upstream.items() if key != "path"}
        != {key: value for key, value in replay_upstream.items() if key != "path"}
        or primary.get("dependency_lock") != replay.get("dependency_lock")
        or primary.get("adapter_source_sha256") != replay.get("adapter_source_sha256")
        or primary.get("canonical_input") != replay.get("canonical_input")
        or primary.get("runner") != replay.get("runner")
    ):
        raise ExternalLeagueError("TooFlashy copied replay frozen identity differs between runs")

    primary_projection = _tooflashy_copied_replay_projection(primary_path, primary)
    replay_projection = _tooflashy_copied_replay_projection(replay_path, replay)
    if primary_projection != replay_projection:
        raise ExternalLeagueError("TooFlashy copied replay dependency closure or result differs")

    primary_roots = _tooflashy_private_storage_roots(primary_path, primary)
    replay_roots = _tooflashy_private_storage_roots(replay_path, replay)
    all_roots = [*primary_roots.values(), *replay_roots.values()]
    if len(set(all_roots)) != len(all_roots):
        raise ExternalLeagueError("TooFlashy copied replay private roots overlap")
    for left_index, left in enumerate(all_roots):
        for right in all_roots[left_index + 1:]:
            if left in right.parents or right in left.parents:
                raise ExternalLeagueError("TooFlashy copied replay private roots are nested")
    primary_storage, primary_identities = _tooflashy_private_storage_manifest(primary_roots)
    replay_storage, replay_identities = _tooflashy_private_storage_manifest(replay_roots)
    shared_identities = primary_identities & replay_identities
    if shared_identities:
        raise ExternalLeagueError("TooFlashy copied replay private storage shares inodes")

    blockers = [
        "detector_scoring_out_of_scope",
        "independent_gold_not_verified",
        "equal_budget_three_repeat_fair_runtime_receipts_missing",
        "external_machine_independence_not_verified",
    ]
    witness: dict[str, object] = {
        "schema": TOOFLASHY_COPIED_REPLAY_WITNESS_SCHEMA,
        "classification": "LOCAL_COPIED_RUNTIME_REPLAY_NOT_SCORING",
        "primary_adapter": {
            "path": str(primary_path),
            "sha256": _sha256_file(primary_path),
            "status": "VERIFIED",
        },
        "replay_adapter": {
            "path": str(replay_path),
            "sha256": _sha256_file(replay_path),
            "status": "VERIFIED",
        },
        "frozen_identity": {
            "upstream": {key: value for key, value in primary_upstream.items() if key != "path"},
            "dependency_lock": primary.get("dependency_lock"),
            "adapter_source_sha256": primary.get("adapter_source_sha256"),
            "canonical_input": primary.get("canonical_input"),
            "runner": primary.get("runner"),
        },
        "replay_agreement": primary_projection,
        "private_storage_independence": {
            "primary": primary_storage,
            "replay": replay_storage,
            "primary_identity_count": len(primary_identities),
            "replay_identity_count": len(replay_identities),
            "shared_inode_count": 0,
        },
        "execution_witness_status": "COPIED_LOCKED_RUNTIME_REPLAY_VERIFIED",
        "status": "VERIFIED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": blockers,
    }
    if destination is not None:
        output = Path(destination).resolve()
        if output.exists():
            raise FileExistsError(f"TooFlashy copied replay witness already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, witness)
        return {**witness, "receipt": str(output)}
    return witness


def _scheduled_tooflashy_adapter_witness_is_bound(
    child_path: Path,
    child: Mapping[str, object],
    *,
    input_path: Path,
    checkout: Path,
) -> bool:
    raw_ref = child.get("raw_output")
    adapter_build = child.get("adapter_build")
    witness_ref = child.get("adapter_execution_witness")
    command = child.get("command")
    comparator = child.get("comparator")
    if (
        child.get("status") != "PROCESS_VALID"
        or child.get("exit_code") != 0
        or child.get("parse_error") is not None
        or not isinstance(raw_ref, Mapping)
        or not isinstance(raw_ref.get("path"), str)
        or raw_ref.get("mode") != "file"
        or not isinstance(adapter_build, Mapping)
        or not isinstance(witness_ref, Mapping)
        or not isinstance(witness_ref.get("path"), str)
        or not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or not isinstance(comparator, Mapping)
    ):
        return False
    root = child_path.parent
    raw_path = (root / str(raw_ref["path"])).resolve()
    witness_path = (root / str(witness_ref["path"])).resolve()
    try:
        raw_path.relative_to(root)
        witness_path.relative_to(root)
    except ValueError:
        return False
    if (
        not raw_path.is_file()
        or raw_ref.get("sha256") != _sha256_file(raw_path)
        or not witness_path.is_file()
        or witness_ref.get("sha256") != _sha256_file(witness_path)
    ):
        return False
    try:
        witness = _load_child_runtime_probe(witness_path)
    except ExternalLeagueError:
        return False
    if witness_ref.get("observation") != witness:
        return False
    uv = Path(str(comparator.get("binary", ""))).resolve()
    adapter_environment = {
        "HOME": str(root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_PROJECT": str(checkout),
        "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
    }
    expected_build = [
        str(uv), "sync", "--locked", "--project", str(checkout),
        "--directory", str(checkout),
    ]
    build_stdout = root / "adapter-build.stdout.bin"
    build_stderr = root / "adapter-build.stderr.bin"
    if (
        set(adapter_build) != {
            "command", "environment", "environment_sha256", "exit_code",
            "stdout_sha256", "stderr_sha256",
        }
        or adapter_build.get("command") != expected_build
        or adapter_build.get("environment") != adapter_environment
        or adapter_build.get("environment_sha256") != _canonical_json_sha256(adapter_environment)
        or adapter_build.get("exit_code") != 0
        or not build_stdout.is_file()
        or not build_stderr.is_file()
        or adapter_build.get("stdout_sha256") != _sha256_file(build_stdout)
        or adapter_build.get("stderr_sha256") != _sha256_file(build_stderr)
    ):
        return False
    adapter_hash = _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
    adapter_tool = [
        str(Path("/usr/bin/env").resolve()), "-i",
        *[f"{key}={value}" for key, value in sorted(adapter_environment.items())],
        str(uv), "run", "--locked", "--no-sync", "--project", str(checkout),
        "--directory", str(checkout), "python", "-c",
        _TOOFLASHY_PARITY_ADAPTER_SCRIPT, str(input_path), str(raw_path), adapter_hash,
    ]
    expected_command = [
        str(Path(sys.executable).resolve()), "-c", _RUNTIME_PROBE_SCRIPT,
        str(witness_path), str(input_path), "-", *adapter_tool,
    ]
    if command != expected_command:
        return False
    stdout_path = root / "stdout.bin"
    stderr_path = root / "stderr.bin"
    if (
        not stdout_path.is_file()
        or not stderr_path.is_file()
        or child.get("stdout_sha256") != _sha256_file(stdout_path)
        or child.get("stderr_sha256") != _sha256_file(stderr_path)
    ):
        return False
    effective = witness.get("effective_environment")
    cache = effective.get("cache") if isinstance(effective, Mapping) else None
    timing = witness.get("child_timing")
    launcher = witness.get("launcher_identity_environment")
    if (
        witness.get("schema") != "flashpatch-l7-child-runtime-probe-v1"
        or not isinstance(cache, Mapping)
        or cache.get("input_sha256") != _sha256_file(input_path)
        or cache.get("input_bytes") != input_path.stat().st_size
        or witness.get("schedule_observation") is not None
        or not isinstance(launcher, Mapping)
        or launcher.get("PWD") is not None
        or launcher.get("UV_PROJECT") != str(checkout)
        or not isinstance(timing, Mapping)
    ):
        return False
    timing_values = [
        timing.get("probe_started_monotonic_ns"),
        timing.get("tool_started_monotonic_ns"),
        timing.get("tool_finished_monotonic_ns"),
    ]
    return (
        all(not isinstance(value, bool) and isinstance(value, int) for value in timing_values)
        and timing_values == sorted(timing_values)
        and timing_values[0] < timing_values[1] < timing_values[2]
    )


def _audit_tooflashy_decoder_timeline(
    child_path: Path,
    child: Mapping[str, object],
    contract: Mapping[str, object],
    adapter_receipt: Path | None = None,
) -> dict[str, object]:
    raw_ref = child.get("raw_output")
    if not isinstance(raw_ref, Mapping) or not isinstance(raw_ref.get("path"), str):
        raise ExternalLeagueError("TooFlashy native JSON output is missing")
    raw_path = _resolve_run_owned_artifact(
        child_path,
        raw_ref["path"],
        label="TooFlashy native JSON",
    )
    if not raw_path.is_file() or raw_ref.get("sha256") != _sha256_file(raw_path):
        raise ExternalLeagueError("TooFlashy native JSON hash mismatch")
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("TooFlashy native JSON cannot be reopened") from exc
    if not isinstance(payload, Mapping):
        raise ExternalLeagueError("TooFlashy native JSON root is invalid")
    input_ref = child.get("input")
    input_path = Path(str(input_ref.get("path", ""))).resolve() if isinstance(input_ref, Mapping) else Path()
    if not input_path.is_file() or input_path != Path(str(contract["canonical_video"]["path"])):
        raise ExternalLeagueError("TooFlashy actual input path differs from canonical input")
    untrusted_augmentation = "decode_audit" in payload
    if adapter_receipt is not None:
        verified = verify_tooflashy_parity_adapter(
            adapter_receipt,
            Path(str(contract["canonical_video"]["path"])),
            Path(str(contract["conversion_receipt"]["path"])),
        )
        canonical_fixture = verified["fixtures"][0] if verified.get("status") == "VERIFIED" else None
        adapter_ref = child.get("tooflashy_parity_adapter")
        adapter_build = child.get("adapter_build")
        child_census = child.get("census_receipt")
        adapter_census = verified.get("census_receipt")
        comparator = child.get("comparator")
        runner = verified.get("runner")
        if (
            canonical_fixture is None
            or payload.get("schema") != "flashpatch-l7-tooflashy-child-adapter-v1"
            or raw_ref.get("mode") != "file"
            or not isinstance(adapter_ref, Mapping)
            or adapter_ref
            != {"path": str(adapter_receipt), "sha256": _sha256_file(adapter_receipt), "status": "VERIFIED"}
            or not isinstance(adapter_build, Mapping)
            or not isinstance(child_census, Mapping)
            or not isinstance(adapter_census, Mapping)
            or child_census.get("path") != adapter_census.get("path")
            or child_census.get("sha256") != adapter_census.get("sha256")
            or not isinstance(comparator, Mapping)
            or comparator.get("revision") != verified["upstream"]["revision"]
            or comparator.get("repository_url") != verified["upstream"]["repository_url"]
            or comparator.get("license") != verified["upstream"]["license"]
            or not isinstance(runner, Mapping)
            or comparator.get("binary_sha256") != runner.get("uv_sha256")
            or comparator.get("working_directory") != verified["upstream"]["path"]
        ):
            raise ExternalLeagueError("scheduled TooFlashy child is not the verified parity adapter")
        adapter_path = _load_tooflashy_adapter_artifact(
            adapter_receipt.parent,
            canonical_fixture["adapter_output"]["path"],
            label="TooFlashy conformance adapter output",
        )
        adapter_cli_path = _load_tooflashy_adapter_artifact(
            adapter_receipt.parent,
            canonical_fixture["official_cli_output"]["path"],
            label="TooFlashy conformance official CLI output",
        )
        conformance_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
        official_payload = json.loads(adapter_cli_path.read_text(encoding="utf-8"))
        decode = payload.get("decode")
        result = payload.get("result")
        public_api = payload.get("public_api")
        runtime = payload.get("runtime")
        conformance_runtime = conformance_payload.get("runtime")
        ledger = decode.get("ledger") if isinstance(decode, Mapping) else None
        canonical_rows = [
            {
                "index": row["frame_index"],
                "cfr_timestamp_us": row["cfr_timestamp_us"],
                "shape": list(contract["shape"][1:]),
                "pixel_format": "rgb24",
                "rgb_sha256": row["rgb_sha256"],
            }
            for row in contract["frame_map"]
        ]
        conversion_payload = _load_conversion_receipt(
            Path(str(contract["conversion_receipt"]["path"])),
            input_path,
        )
        materializer = conversion_payload.get("materializer")
        canonical_ffmpeg = materializer.get("binary_sha256") if isinstance(materializer, Mapping) else None
        adapter_environment = adapter_build.get("environment")
        if (
            not isinstance(decode, Mapping)
            or ledger != canonical_rows
            or decode.get("ledger_sha256") != _canonical_json_sha256(ledger)
            or not isinstance(result, Mapping)
            or {field: result.get(field) for field in TOOFLASHY_OFFICIAL_JSON_FIELDS}
            != official_payload
            or result.get("event_representation")
            != {"event_count": official_payload.get("event_count"), "failures": official_payload.get("failures")}
            or public_api != conformance_payload.get("public_api")
            or not isinstance(runtime, Mapping)
            or not isinstance(conformance_runtime, Mapping)
            or any(
                runtime.get(field) != conformance_runtime.get(field)
                for field in (
                    "python_executable",
                    "python_executable_sha256",
                    "python_version",
                    "dependency_versions",
                    "dependency_evidence",
                    "decoder_executables",
                )
            )
            or not isinstance(adapter_environment, Mapping)
            or not isinstance(canonical_ffmpeg, str)
            or not _tooflashy_child_runtime_is_frozen(
                runtime,
                parent_environment=adapter_environment,
                checkout=Path(str(verified["upstream"]["path"])),
                root=child_path.parent,
                canonical_ffmpeg_sha256=canonical_ffmpeg,
            )
            or not _scheduled_tooflashy_adapter_witness_is_bound(
                child_path,
                child,
                input_path=input_path,
                checkout=Path(str(verified["upstream"]["path"])),
            )
        ):
            raise ExternalLeagueError("scheduled TooFlashy adapter ledger, result, API, or runtime drifted")
        return {
            "parity_status": "VERIFIED",
            "parity_reason": None,
            "native_artifacts": [
                {"path": str(raw_path), "sha256": _sha256_file(raw_path)},
                {"path": str(adapter_path), "sha256": _sha256_file(adapter_path)},
                {"path": str(adapter_receipt), "sha256": _sha256_file(adapter_receipt)},
            ],
            "actual_input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
            "reported_frame_count": decode["frame_count"],
            "reported_fps": decode["fps"],
            "frame_index_base": 0,
            "timestamp_precision_us": 1,
            "max_timestamp_alignment_error_us": 0,
            "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
            "canonical_frame_map_sha256": contract["frame_map_sha256"],
            "frame_map_relation": "EXACT_ZERO_BASED_SCHEDULED_CHILD_PRECONSUMPTION_LEDGER",
            "native_frame_map_sha256": _canonical_json_sha256(ledger),
            "official_cli_conformance": "VERIFIED",
            "local_contract_status": "VERIFIER_REPLAY_CONFIRMED",
            "execution_witness_status": "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT",
            "comparison_eligible": False,
            "league_child_binding": "LOCAL_RECEIPT_CONSISTENT_SAME_PROCESS_EVIDENCE",
        }
    return {
        "parity_status": "NOT_VERIFIED",
        "parity_reason": (
            "untrusted_unfrozen_decode_audit_augmentation"
            if untrusted_augmentation
            else "native_json_omits_per_frame_decode_identity_and_timestamps"
        ),
        "native_artifacts": [{"path": str(raw_path), "sha256": _sha256_file(raw_path)}],
        "actual_input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
        "reported_frame_count": payload.get("frame_count"),
        "reported_fps": payload.get("fps"),
        "frame_index_base": None,
        "timestamp_precision_us": None,
        "max_timestamp_alignment_error_us": None,
        "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
        "canonical_frame_map_sha256": contract["frame_map_sha256"],
        "frame_map_relation": "UNAVAILABLE_FROM_PINNED_NATIVE_OUTPUT",
        "native_frame_map_sha256": None,
    }


def _audit_flashpatch_decoder_timeline(
    child_path: Path,
    child: Mapping[str, object],
    video: Path,
    conversion: Path,
    contract: Mapping[str, object],
) -> dict[str, object]:
    decode_ref = child.get("worker_decode")
    if not isinstance(decode_ref, Mapping) or not isinstance(decode_ref.get("path"), str):
        raise ExternalLeagueError("FlashPatch worker decoder audit is missing")
    decode_path = _resolve_run_owned_artifact(
        child_path,
        decode_ref["path"],
        label="FlashPatch worker decoder audit",
    )
    if not decode_path.is_file() or decode_ref.get("sha256") != _sha256_file(decode_path):
        raise ExternalLeagueError("FlashPatch worker decoder audit hash mismatch")
    try:
        stored_decode = json.loads(decode_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("FlashPatch worker decoder audit cannot be reopened") from exc
    _, _, fresh_decode = _decode_canonical_video_rgb(video, conversion)
    if stored_decode != fresh_decode or decode_ref.get("receipt") != fresh_decode:
        raise ExternalLeagueError("FlashPatch worker decoder audit differs from fresh canonical replay")
    return {
        "parity_status": "VERIFIED",
        "parity_reason": None,
        "native_artifacts": [{"path": str(decode_path), "sha256": _sha256_file(decode_path)}],
        "actual_input": {"path": str(video), "sha256": _sha256_file(video)},
        "decoded_frame_count": contract["frame_count"],
        "fps": contract["fps"],
        "frame_index_base": 0,
        "timestamp_precision_us": 1,
        "max_timestamp_alignment_error_us": 0,
        "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
        "canonical_frame_map_sha256": contract["frame_map_sha256"],
        "frame_map_relation": "EXACT_ZERO_BASED_FRESH_REPLAY",
        "native_frame_map_sha256": contract["frame_map_sha256"],
    }


def _audit_iris_source_decoder_timeline(
    child_path: Path,
    child: Mapping[str, object],
    video: Path,
    conversion: Path,
    contract: Mapping[str, object],
    conformance_receipt: Path | None,
    semantic_receipt: Path | None,
) -> dict[str, object]:
    if conformance_receipt is None:
        raise ExternalLeagueError("EA IRIS source adapter conformance receipt is required")
    if semantic_receipt is None:
        raise ExternalLeagueError("EA IRIS source adapter semantic conformance receipt is required")
    run, raw, observation, reopened_path = _load_iris_source_adapter_run(
        child_path,
        expected_lane="DIRECT_DETECTOR",
    )
    conformance, conformance_path = _load_iris_source_conformance_receipt(conformance_receipt)
    semantic, semantic_path = _load_iris_realtime_semantic_probe(semantic_receipt)
    input_ref = run.get("input")
    conversion_ref = run.get("conversion_receipt")
    build_ref = run.get("build_receipt")
    if (
        reopened_path != child_path.resolve()
        or not isinstance(input_ref, Mapping)
        or Path(str(input_ref.get("path", ""))).resolve() != video
        or input_ref.get("sha256") != contract["canonical_video"]["sha256"]
        or not isinstance(conversion_ref, Mapping)
        or Path(str(conversion_ref.get("path", ""))).resolve() != conversion
        or conversion_ref.get("sha256") != contract["conversion_receipt"]["sha256"]
        or not isinstance(build_ref, Mapping)
        or conformance.get("source_build") != build_ref
        or conformance.get("status") != "LOCAL_CONFORMANCE_MATCH"
        or conformance.get("local_fixture_match") is not True
        or semantic.get("status") != "SEMANTIC_MISMATCH_NOT_VERIFIED"
        or semantic.get("direct_participant_authorized") is not False
        or semantic.get("scoreable") is not False
    ):
        raise ExternalLeagueError("EA IRIS source adapter run differs from canonical input or conformance authority")
    frames = raw.get("frames")
    if not isinstance(frames, list) or len(frames) != int(contract["frame_count"]):
        raise ExternalLeagueError("EA IRIS source adapter native frame ledger is incomplete")
    native_map = [
        {
            "frame_index": row["frame_index"],
            "cfr_timestamp_us": row["cfr_timestamp_us_rounded"],
            "renderer_timestamp_us": row["renderer_timestamp_us"],
            "rgb_sha256": row["rgb_sha256"],
        }
        for row in frames
    ]
    if native_map != contract["frame_map"]:
        raise ExternalLeagueError("EA IRIS source adapter native RGB/timeline ledger differs from canonical map")
    semantic_build = semantic["source_build"]
    run_build_path = Path(str(build_ref["path"])).resolve()
    run_build, run_build_path = _load_iris_source_build_receipt(run_build_path)
    if (
        Path(str(semantic_build["path"])).resolve() != run_build_path
        or semantic_build["sha256"] != build_ref["sha256"]
        or semantic_build["direct_binary_sha256"] != run_build["binaries"]["source_frame_adapter"]["sha256"]
        or semantic_build["source_video_binary_sha256"] != run_build["binaries"]["source_video_oracle"]["sha256"]
    ):
        raise ExternalLeagueError("EA IRIS semantic mismatch authority differs from direct run build")
    return {
        "parity_status": "NOT_VERIFIED",
        "parity_reason": "realtime_and_source_video_frame_category_timing_mismatch",
        "native_artifacts": [
            {"path": str(child_path), "sha256": _sha256_file(child_path)},
            {"path": str(conformance_path), "sha256": _sha256_file(conformance_path)},
            {"path": str(semantic_path), "sha256": _sha256_file(semantic_path)},
        ],
        "actual_input": {"path": str(video), "sha256": _sha256_file(video)},
        "decoded_frame_count": contract["frame_count"],
        "fps": contract["fps"],
        "frame_index_base": 0,
        "timestamp_precision_us": 1,
        "max_timestamp_alignment_error_us": 0,
        "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
        "canonical_frame_map_sha256": contract["frame_map_sha256"],
        "frame_map_relation": "EXACT_ZERO_BASED_ANALYSEFRAME_BOUNDARY_LEDGER",
        "native_frame_map_sha256": _canonical_json_sha256(native_map),
        "source_release_conformance": "SEMANTIC_MISMATCH_NOT_VERIFIED",
        "semantic_conformance": {
            "status": semantic["status"],
            "frame_categories_exact": semantic["comparison"]["frame_categories_exact"],
            "terminal_agreement": semantic["comparison"]["terminal_agreement"],
            "normalization_applied": semantic["comparison"]["normalization_applied"],
        },
        "local_contract_status": "VERIFIER_REPLAY_CONFIRMED",
        "execution_witness_status": "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT",
        "comparison_eligible": False,
        "runtime_timing_eligible": observation.get("runtime_timing_eligible") is True,
    }


def verify_decoder_timeline_parity(
    run_receipts: Sequence[Path | str],
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    *,
    tooflashy_adapter_receipt: Path | str | None = None,
    iris_source_conformance_receipt: Path | str | None = None,
    iris_semantic_conformance_receipt: Path | str | None = None,
    destination: Path | str | None = None,
) -> dict[str, object]:
    """Verify primary case-level input parity without minting a league score.

    Each child receipt is reopened together with its native raw decode output.
    Interval evidence is reported separately because a case-level detector may
    legitimately expose no onset or interval endpoint.
    """
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    failures: list[str] = []
    parity_blockers: list[str] = []
    rows: list[dict[str, object]] = []
    try:
        _, contract = _canonical_decoder_timeline_contract(video, conversion)
    except (ExternalLeagueError, OSError, ValueError) as exc:
        contract = {
            "canonical_video": {"path": str(video), "sha256": None},
            "conversion_receipt": {"path": str(conversion), "sha256": None},
            "frame_map": [],
            "frame_map_sha256": None,
        }
        failures.append(f"canonical_decoder_contract_invalid:{exc}")
    refs_valid = isinstance(run_receipts, Sequence) and not isinstance(run_receipts, (str, bytes))
    refs = run_receipts if refs_valid else ()
    if not refs_valid:
        failures.append("run_receipt_collection_invalid")
    loaded: dict[str, tuple[Path, Mapping[str, object], str]] = {}
    for ref in refs:
        if not isinstance(ref, (str, os.PathLike)):
            failures.append("run_receipt_reference_invalid")
            continue
        path = Path(ref).resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"run_receipt_unreadable:{path}")
            continue
        comparator = payload.get("comparator") if isinstance(payload, Mapping) else None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema") == KAYA_NATURAL_CASE_PARITY_SCHEMA
            and payload.get("identity") == KAYA_DIRECT_PARTICIPANT_ID
        ):
            name = KAYA_DIRECT_PARTICIPANT_ID
            receipt_kind = "kaya_natural_case_parity"
        else:
            name = comparator.get("name") if isinstance(comparator, Mapping) else None
            receipt_kind = "process_run"
        if name not in DIRECT_DETECTOR_POPULATION:
            failures.append(f"run_receipt_comparator_invalid:{path}")
            continue
        if (
            (name == KAYA_DIRECT_PARTICIPANT_ID and receipt_kind != "kaya_natural_case_parity")
            or (name != KAYA_DIRECT_PARTICIPANT_ID and receipt_kind != "process_run")
        ):
            failures.append(f"run_receipt_kind_invalid:{name}:{path}")
            continue
        if name in loaded:
            failures.append(f"duplicate_run_receipt:{name}")
            continue
        loaded[str(name)] = (path, payload, receipt_kind)
    missing = [name for name in DIRECT_DETECTOR_POPULATION if name not in loaded]
    failures.extend(f"run_receipt_missing:{name}" for name in missing)
    for name in DIRECT_DETECTOR_POPULATION:
        if name not in loaded:
            continue
        path, child, receipt_kind = loaded[name]
        row: dict[str, object] = {
            "comparator": name,
            "run_receipt": {"path": str(path), "sha256": _sha256_file(path)},
            "status": "NOT_VERIFIED",
            "primary_case_level_endpoint": {
                "status": "NOT_VERIFIED",
                "reason": "decoder_timeline_parity_unverified",
            },
            "secondary_interval_endpoint": {
                "status": "NOT_VERIFIED",
                "reason": "decoder_timeline_parity_unverified",
            },
        }
        try:
            if failures and contract.get("frame_map_sha256") is None:
                raise ExternalLeagueError("canonical decoder contract is unavailable")
            if name == KAYA_DIRECT_PARTICIPANT_ID:
                if receipt_kind != "kaya_natural_case_parity":
                    raise ExternalLeagueError("Kaya must use its natural-case parity receipt")
                verified_kaya = verify_kaya_natural_case_parity_receipt(path)
                if (
                    verified_kaya.get("status") != "VERIFIED"
                    or verified_kaya.get("claim_status") != "NOT_SCOREABLE"
                    or verified_kaya.get("scoreable") is not False
                    or verified_kaya.get("comparison_eligible") is not False
                    or verified_kaya.get("canonical_contract") != contract
                ):
                    raise ExternalLeagueError(
                        "Kaya natural-case receipt differs from the canonical claim boundary"
                    )
                native_output = verified_kaya.get("native_output")
                results = native_output.get("results") if isinstance(native_output, Mapping) else None
                if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
                    raise ExternalLeagueError("Kaya natural-case native result is invalid")
                intervals = results[0].get("interval_tuples")
                if not isinstance(intervals, list):
                    raise ExternalLeagueError("Kaya natural-case interval result is invalid")
                hazard_frame_indices = sorted({
                    index
                    for interval in intervals
                    if isinstance(interval, list) and len(interval) == 3
                    for index in range(interval[1], interval[2])
                })
                observation = {
                    "prediction": "HAZARDOUS" if intervals else "SAFE",
                    "hazard_frame_indices": hazard_frame_indices,
                }
                native_capture = native_output.get("capture_runs")
                native_ledger = (
                    native_capture[0].get("ledger")
                    if isinstance(native_capture, list)
                    and len(native_capture) == 1
                    and isinstance(native_capture[0], Mapping)
                    else None
                )
                if not isinstance(native_ledger, list):
                    raise ExternalLeagueError("Kaya natural-case native decode ledger is invalid")
                decoder = {
                    "parity_status": "VERIFIED",
                    "parity_reason": None,
                    "native_artifacts": [{"path": str(path), "sha256": _sha256_file(path)}],
                    "actual_input": {
                        "path": str(video),
                        "sha256": contract["canonical_video"]["sha256"],
                    },
                    "decoded_frame_count": contract["frame_count"],
                    "fps": contract["fps"],
                    "frame_index_base": 0,
                    "timestamp_precision_us": 1,
                    "max_timestamp_alignment_error_us": 0,
                    "conversion_receipt_sha256": contract["conversion_receipt"]["sha256"],
                    "canonical_frame_map_sha256": contract["frame_map_sha256"],
                    "frame_map_relation": "EXACT_ZERO_BASED_UPSTREAM_PRECONSUMPTION_BGR_LEDGER",
                    "native_frame_map_sha256": _canonical_json_sha256(native_ledger),
                    "local_contract_status": "NATIVE_DIRECT_EXACT_RECEIPT_REOPENED",
                    "execution_witness_status": "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT",
                    "comparison_eligible": False,
                }
            else:
                if receipt_kind != "process_run":
                    raise ExternalLeagueError("detector process receipt kind is invalid")
                conversion_ref = child.get("conversion_receipt")
                if (
                    not isinstance(conversion_ref, Mapping)
                    or not isinstance(conversion_ref.get("path"), str)
                    or Path(str(conversion_ref["path"])).resolve() != conversion
                    or conversion_ref.get("sha256") != contract["conversion_receipt"]["sha256"]
                ):
                    raise ExternalLeagueError("child conversion binding differs from canonical receipt")
                if child.get("status") != "PROCESS_VALID":
                    raise ExternalLeagueError("child process receipt is not PROCESS_VALID")
                observation = _reopen_child_normalized_observation(
                    path,
                    child,
                    name,
                    str(contract["canonical_video"]["sha256"]),
                )
                stored_observation = child.get("observation") if name == "FlashPatch" else child.get("parsed_observation")
                if not isinstance(stored_observation, Mapping) or _canonical_json_sha256(stored_observation) != _canonical_json_sha256(observation):
                    raise ExternalLeagueError("stored normalized observation differs from reopened native output")
                if name == "FlashPatch":
                    decoder = _audit_flashpatch_decoder_timeline(path, child, video, conversion, contract)
                elif name == "TooFlashy":
                    decoder = _audit_tooflashy_decoder_timeline(
                        path,
                        child,
                        contract,
                        Path(tooflashy_adapter_receipt).resolve()
                        if tooflashy_adapter_receipt is not None
                        else None,
                    )
                else:
                    raise ExternalLeagueError("detector participant has no decoder parity route")
            prediction = observation.get("prediction")
            if prediction not in {"SAFE", "HAZARDOUS"}:
                raise ExternalLeagueError("native observation lacks a case-level endpoint")
            decoder_verified = decoder.get("parity_status") == "VERIFIED"
            comparison_eligible = decoder_verified and decoder.get("comparison_eligible", True) is True
            if not decoder_verified:
                parity_blockers.append(
                    f"decoder_parity_unverified:{name}:{decoder.get('parity_reason', 'reason_unavailable')}"
                )
            elif not comparison_eligible:
                parity_blockers.append(f"decoder_comparison_ineligible:{name}:independent_execution_witness_missing")
            interval = observation.get("hazard_frame_indices")
            if name == "TooFlashy":
                secondary = {
                    "status": "NOT_VERIFIED",
                    "reason": "native_tool_does_not_expose_interval_endpoint",
                    "value": None,
                }
            elif not decoder_verified:
                secondary = {
                    "status": "NOT_VERIFIED",
                    "reason": "native_interval_not_bound_to_canonical_rgb_identity",
                    "value": None,
                }
            elif (
                not isinstance(interval, list)
                or any(isinstance(index, bool) or not isinstance(index, int) for index in interval)
                or interval != sorted(set(interval))
                or any(index < 0 or index >= int(contract["frame_count"]) for index in interval)
            ):
                secondary = {
                    "status": "NOT_VERIFIED",
                    "reason": "native_interval_mapping_invalid",
                    "value": None,
                }
            else:
                secondary = {
                    "status": "VERIFIED",
                    "unit": "canonical_frame_index",
                    "hazard_frame_indices": interval,
                    "timestamp_precision_us": decoder["timestamp_precision_us"],
                }
            row.update({
                "status": "VERIFIED" if decoder_verified else "NOT_VERIFIED",
                "decoder_timeline": decoder,
                "primary_case_level_endpoint": {
                    "status": "VERIFIED" if decoder_verified else "NOT_VERIFIED",
                    "prediction": prediction if decoder_verified else None,
                    "reason": None if decoder_verified else "decoder_timeline_parity_unverified",
                    "comparison_eligible": comparison_eligible,
                },
                "secondary_interval_endpoint": secondary,
            })
        except (ExternalLeagueError, OSError, ValueError, KeyError) as exc:
            failure = f"decoder_timeline_evidence_invalid:{name}:{exc}"
            failures.append(failure)
            row["failure"] = failure
        rows.append(row)
    primary_verified = (
        not failures
        and len(rows) == len(DIRECT_DETECTOR_POPULATION)
        and all(row.get("status") == "VERIFIED" for row in rows)
        and all(row["primary_case_level_endpoint"].get("status") == "VERIFIED" for row in rows)
        and all(row["primary_case_level_endpoint"].get("comparison_eligible") is True for row in rows)
    )
    secondary_verified = primary_verified and all(
        row["secondary_interval_endpoint"].get("status") == "VERIFIED"
        for row in rows
    )
    receipt: dict[str, object] = {
        "schema": DECODER_TIMELINE_PARITY_SCHEMA,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "canonical_contract": contract,
        "comparators": rows,
        "primary_case_level_comparison": "VERIFIED" if primary_verified else "NOT_VERIFIED",
        "secondary_interval_comparison": "VERIFIED" if secondary_verified else "NOT_VERIFIED",
        "status": "PRIMARY_CASE_PARITY_VERIFIED" if primary_verified else "NOT_VERIFIED",
        "failures": failures,
        "parity_blockers": parity_blockers,
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "three_repeat_bundle_not_verified_by_this_gate",
            "fair_runtime_bundle_not_verified_by_this_gate",
            "independent_gold_not_verified_by_this_gate",
            "frozen_public_case_ledger_not_verified_by_this_gate",
        ],
    }
    if destination is not None:
        destination_path = Path(destination).resolve()
        if destination_path.exists():
            raise FileExistsError(f"decoder timeline parity receipt already exists: {destination_path}")
        _write_json(destination_path, receipt)
        receipt = {**receipt, "receipt": str(destination_path)}
    return receipt


def _observed_environment_matches_protocol(
    observed: object,
    protocol: Mapping[str, object],
    input_sha256: str,
    schedule_binding: Mapping[str, object] | None,
) -> bool:
    if not isinstance(observed, Mapping):
        return False
    parent = observed.get("parent_precondition")
    child = observed.get("child_probe")
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        return False
    machine = parent.get("machine")
    cpu = parent.get("cpu")
    gpu = parent.get("gpu")
    cache = parent.get("cache")
    concurrency = parent.get("concurrency")
    taskset = cpu.get("taskset") if isinstance(cpu, Mapping) else None
    bubblewrap = gpu.get("bubblewrap") if isinstance(gpu, Mapping) else None
    docker = gpu.get("docker") if isinstance(gpu, Mapping) else None
    isolation = protocol.get("gpu", {}).get("isolation") if isinstance(protocol.get("gpu"), Mapping) else None
    isolation_valid = (
        isolation == "BWRAP_EMPTY_DEV" and isinstance(bubblewrap, Mapping)
    ) or (
        isolation == "DOCKER_EMPTY_DEV"
        and isinstance(docker, Mapping)
        and set(docker) == {"container_marker_sha256", "visible_device_nodes"}
        and isinstance(docker.get("container_marker_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", docker["container_marker_sha256"]) is not None
    )
    if not all(isinstance(item, Mapping) for item in (machine, cpu, gpu, cache, concurrency, taskset)) or not isolation_valid:
        return False
    taskset_path = Path(str(taskset.get("path", ""))).resolve()
    bubblewrap_path = Path(str(bubblewrap.get("path", ""))).resolve() if isinstance(bubblewrap, Mapping) else None
    available_affinity = cpu.get("available_affinity")
    if (
        not isinstance(available_affinity, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in available_affinity)
    ):
        return False
    effective_environment = child.get("effective_environment")
    if not isinstance(effective_environment, Mapping):
        return False
    expected_resource_environment = _runtime_policy_environment(protocol)
    expected_process_environment = _canonical_fair_base_environment(None)
    expected_process_environment.update(expected_resource_environment)
    input_bytes = cache.get("input_bytes")
    if isinstance(input_bytes, bool) or not isinstance(input_bytes, int) or input_bytes <= 0:
        return False
    child_gpu = effective_environment.get("gpu")
    visible_device_nodes = child_gpu.get("visible_device_nodes") if isinstance(child_gpu, Mapping) else None
    allowed_device_nodes = {
        "/dev/console",
        "/dev/full",
        "/dev/null",
        "/dev/ptmx",
        "/dev/pts/ptmx",
        "/dev/random",
        "/dev/tty",
        "/dev/urandom",
        "/dev/zero",
    }
    if (
        not isinstance(visible_device_nodes, list)
        or not all(isinstance(path, str) and path.startswith("/dev/") for path in visible_device_nodes)
        or any(
            path not in allowed_device_nodes and re.fullmatch(r"/dev/pts/[0-9]+", path) is None
            for path in visible_device_nodes
        )
        or not {"/dev/null", "/dev/random", "/dev/urandom", "/dev/zero"}.issubset(visible_device_nodes)
    ):
        return False
    expected_effective_environment = {
        "process_environment": expected_process_environment,
        "machine": protocol["machine"],
        "cpu": {
            "model": protocol["cpu"]["model"],
            "logical_count": protocol["cpu"]["logical_count"],
            "affinity": protocol["cpu"]["affinity"],
        },
        "gpu": {
            "visible_device_nodes": visible_device_nodes,
        },
        "cache": {
            "policy": "WARM_INPUT_PRETOUCHED",
            "input_sha256": input_sha256,
            "input_bytes": input_bytes,
        },
    }
    expected_schedule_observation = None
    if schedule_binding is not None:
        expected_binding_fields = {
            "path",
            "artifact_sha256",
            "schedule_sha256",
            "stat",
            "slot",
            "round",
            "position",
            "comparator",
            "repeat_ordinal",
        }
        if set(schedule_binding) != expected_binding_fields or not isinstance(schedule_binding.get("stat"), Mapping):
            return False
        schedule_path = Path(str(schedule_binding.get("path", ""))).resolve()
        current_stat = schedule_path.stat() if schedule_path.is_file() else None
        expected_stat = None if current_stat is None else {
            "device": current_stat.st_dev,
            "inode": current_stat.st_ino,
            "size": current_stat.st_size,
            "mtime_ns": current_stat.st_mtime_ns,
            "ctime_ns": current_stat.st_ctime_ns,
        }
        if (
            not schedule_path.is_file()
            or schedule_binding.get("artifact_sha256") != _sha256_file(schedule_path)
            or schedule_binding.get("stat") != expected_stat
        ):
            return False
        expected_schedule_observation = {
            "path": str(schedule_path),
            "artifact_sha256": schedule_binding["artifact_sha256"],
            "stat": schedule_binding["stat"],
            "schedule_sha256": schedule_binding["schedule_sha256"],
            "slot": str(schedule_binding["slot"]),
            "round": str(schedule_binding["round"]),
            "position": str(schedule_binding["position"]),
            "comparator": schedule_binding["comparator"],
            "repeat_ordinal": str(schedule_binding["repeat_ordinal"]),
        }
    child_timing = child.get("child_timing")
    launcher_identity = child.get("launcher_identity_environment")
    if (
        not isinstance(launcher_identity, Mapping)
        or set(launcher_identity) != {"PWD", "UV_PROJECT"}
        or not isinstance(launcher_identity.get("PWD"), str)
        or not launcher_identity.get("PWD")
        or (
            launcher_identity.get("UV_PROJECT") is not None
            and not isinstance(launcher_identity.get("UV_PROJECT"), str)
        )
    ):
        return False
    if not isinstance(child_timing, Mapping) or set(child_timing) != {
        "probe_started_monotonic_ns",
        "probe_started_wall_time_ns",
        "tool_started_monotonic_ns",
        "tool_finished_monotonic_ns",
    }:
        return False
    timing_values = list(child_timing.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in timing_values):
        return False
    if not (
        child_timing["probe_started_monotonic_ns"]
        <= child_timing["tool_started_monotonic_ns"]
        < child_timing["tool_finished_monotonic_ns"]
    ):
        return False
    if expected_schedule_observation is not None:
        schedule_stat = expected_schedule_observation["stat"]
        if max(schedule_stat["mtime_ns"], schedule_stat["ctime_ns"]) > child_timing["probe_started_wall_time_ns"]:
            return False
    return (
        machine == protocol["machine"]
        and cpu.get("model") == protocol["cpu"]["model"]
        and cpu.get("logical_count") == protocol["cpu"]["logical_count"]
        and set(protocol["cpu"]["affinity"]).issubset(available_affinity)
        and taskset_path.is_file()
        and taskset.get("sha256") == _sha256_file(taskset_path)
        and gpu.get("policy") == "DISABLED"
        and gpu.get("isolation") == isolation
        and (
            (
                isolation == "BWRAP_EMPTY_DEV"
                and bubblewrap_path is not None
                and bubblewrap_path.is_file()
                and isinstance(bubblewrap, Mapping)
                and bubblewrap.get("sha256") == _sha256_file(bubblewrap_path)
            )
            or (
                isolation == "DOCKER_EMPTY_DEV"
                and isinstance(docker, Mapping)
                and docker.get("visible_device_nodes") == visible_device_nodes
            )
        )
        and cache == {
            "policy": "WARM_INPUT_PRETOUCHED",
            "input_sha256": input_sha256,
            "input_bytes": input_bytes,
        }
        and concurrency == {
            "limit": 1,
            "lock_path": protocol["concurrency"]["lock_path"],
            "lock_acquired": True,
        }
        and child.get("schema") == "flashpatch-l7-child-runtime-probe-v1"
        and child.get("machine") == protocol["machine"]
        and child.get("cpu_affinity") == protocol["cpu"]["affinity"]
        and child.get("resource_environment") == expected_resource_environment
        and child.get("effective_environment_policy_sha256")
        == _canonical_json_sha256(FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY)
        and dict(effective_environment) == expected_effective_environment
        and child.get("effective_environment_sha256")
        == _canonical_json_sha256(expected_effective_environment)
        and isinstance(child.get("full_environment_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", child["full_environment_sha256"]) is not None
        and child.get("schedule_observation") == expected_schedule_observation
    )


def _child_probe_and_command_are_bound(
    child_path: Path,
    child: Mapping[str, object],
    runtime: Mapping[str, object],
    protocol: Mapping[str, object],
    comparator: str,
) -> bool:
    probe_ref = child.get("runtime_probe")
    command = child.get("command")
    observed = runtime.get("observed_environment")
    if (
        not isinstance(probe_ref, Mapping)
        or not isinstance(probe_ref.get("path"), str)
        or not isinstance(command, list)
        or not all(isinstance(part, str) for part in command)
        or len(command) < 8
        or not isinstance(observed, Mapping)
    ):
        return False
    probe_path = child_path.parent / str(probe_ref["path"])
    try:
        probe = _load_child_runtime_probe(probe_path)
    except ExternalLeagueError:
        return False
    parent = observed.get("parent_precondition")
    if not isinstance(parent, Mapping) or not isinstance(parent.get("cpu"), Mapping):
        return False
    taskset = parent["cpu"].get("taskset")
    gpu = parent.get("gpu")
    bubblewrap = gpu.get("bubblewrap") if isinstance(gpu, Mapping) else None
    isolation = protocol.get("gpu", {}).get("isolation") if isinstance(protocol.get("gpu"), Mapping) else None
    if not isinstance(taskset, Mapping) or (
        isolation == "BWRAP_EMPTY_DEV" and not isinstance(bubblewrap, Mapping)
    ) or isolation not in {"BWRAP_EMPTY_DEV", "DOCKER_EMPTY_DEV"}:
        return False
    expected_affinity = ",".join(str(cpu) for cpu in protocol["cpu"]["affinity"])
    conversion_ref = child.get("conversion_receipt")
    if not isinstance(conversion_ref, Mapping) or not isinstance(conversion_ref.get("path"), str):
        return False
    conversion = Path(str(conversion_ref["path"])).resolve()
    input_payload = child.get("input")
    if comparator == "TooFlashy":
        if not isinstance(input_payload, Mapping) or not isinstance(input_payload.get("path"), str):
            return False
        canonical_input = Path(str(input_payload["path"])).resolve()
    else:
        canonical_input = (conversion.parent / "canonical.ffv1.mkv").resolve()
    schedule_binding = runtime.get("schedule_binding")
    if schedule_binding is not None and (
        not isinstance(schedule_binding, Mapping)
        or not isinstance(schedule_binding.get("path"), str)
    ):
        return False
    schedule_argument = str(schedule_binding["path"]) if isinstance(schedule_binding, Mapping) else "-"
    expected_prefix = [str(taskset.get("path")), "--cpu-list", expected_affinity]
    if isolation == "BWRAP_EMPTY_DEV":
        expected_prefix.extend([
            str(bubblewrap.get("path")), "--bind", "/", "/", "--dev", "/dev",
            "--proc", "/proc", "--die-with-parent",
        ])
    expected_prefix.extend([
        str(Path(sys.executable).resolve()),
        "-c",
        _RUNTIME_PROBE_SCRIPT,
        str(probe_path.resolve()),
        str(canonical_input),
        schedule_argument,
    ])
    prefix_valid = (
        probe_ref.get("sha256") == _sha256_file(probe_path)
        and probe_ref.get("observation") == probe
        and observed.get("child_probe") == probe
        and command[: len(expected_prefix)] == expected_prefix
    )
    if not prefix_valid:
        return False
    comparator_payload = child.get("comparator")
    if not isinstance(comparator_payload, Mapping):
        return False
    launcher_identity = probe.get("launcher_identity_environment")
    if not isinstance(launcher_identity, Mapping):
        return False
    tool_command = command[len(expected_prefix):]
    if comparator == KAYA_DIRECT_PARTICIPANT_ID:
        raw_output = child.get("raw_output")
        source_checkout = comparator_payload.get("source_checkout")
        binary_value = comparator_payload.get("binary")
        if (
            not isinstance(input_payload, Mapping)
            or not isinstance(raw_output, Mapping)
            or not isinstance(raw_output.get("path"), str)
            or not isinstance(source_checkout, str)
            or not isinstance(binary_value, str)
        ):
            return False
        python = Path(binary_value).absolute()
        resolved_python = python.resolve()
        checkout = Path(source_checkout).resolve()
        raw_path = (child_path.parent / str(raw_output["path"])).resolve()
        adapter_hash = _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
        expected = [
            str(python), "-S", "-X",
            f"pycache_prefix={child_path.parent / 'pycache'}", "-c",
            _KAYA_CONFORMANCE_CHILD_SCRIPT, "native", str(checkout),
            str(canonical_input), "-", str(raw_path), adapter_hash, "1",
        ]
        return (
            python.is_file()
            and python.parent.name == "bin"
            and resolved_python.is_file()
            and comparator_payload.get("binary_sha256") == _sha256_file(resolved_python)
            and comparator_payload.get("revision") == KAYA_SOURCE_REVISION
            and comparator_payload.get("tree") == KAYA_SOURCE_TREE
            and comparator_payload.get("repository_url") == KAYA_REPOSITORY_URL
            and comparator_payload.get("license") == "BSD-3-Clause"
            and comparator_payload.get("working_directory") == str(child_path.parent.resolve())
            and input_payload.get("path") == str(canonical_input)
            and input_payload.get("sha256") == _sha256_file(canonical_input)
            and launcher_identity == {
                "PWD": str(child_path.parent.resolve()),
                "UV_PROJECT": None,
            }
            and tool_command == expected
        )
    if comparator == "TooFlashy":
        census_ref = child.get("census_receipt")
        raw_output = child.get("raw_output")
        if (
            not isinstance(census_ref, Mapping)
            or not isinstance(census_ref.get("path"), str)
            or not isinstance(input_payload, Mapping)
            or not isinstance(input_payload.get("path"), str)
            or not isinstance(raw_output, Mapping)
            or not isinstance(raw_output.get("path"), str)
        ):
            return False
        census_path = Path(str(census_ref["path"])).resolve()
        try:
            census_payload = json.loads(census_path.read_text(encoding="utf-8"))
            artifact_root_value = census_payload.get("artifact_root")
            if not isinstance(artifact_root_value, str):
                return False
            entry, _ = _load_execution_census_entry(
                census_path,
                artifact_root_value,
                "TooFlashy",
            )
            command_artifact = _resolve_census_artifact(
                Path(artifact_root_value).resolve(),
                entry["command_artifact"],
                name="TooFlashy",
                field="command_artifact",
            )
            template = json.loads(command_artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ExternalLeagueError):
            return False
        if not isinstance(template, list) or not all(isinstance(part, str) for part in template):
            return False
        expected = [
            part.format(
                input=str(Path(str(input_payload["path"])).resolve()),
                output=str((child_path.parent / str(raw_output["path"])).resolve()),
            )
            for part in template
        ]
        executable_name = expected[0]
        executable_value = (
            shutil.which(executable_name)
            if Path(executable_name).name == executable_name
            else executable_name
        )
        if executable_value is None:
            return False
        executable = Path(executable_value).resolve()
        expected[0] = str(executable)
        expected_working_directory = str(Path(str(entry["source_checkout"])).resolve())
        adapter_ref = child.get("tooflashy_parity_adapter")
        adapter_build = child.get("adapter_build")
        if adapter_ref is not None:
            if (
                not isinstance(adapter_ref, Mapping)
                or set(adapter_ref) != {"path", "sha256", "status"}
                or not isinstance(adapter_ref.get("path"), str)
                or adapter_ref.get("status") != "VERIFIED"
                or not isinstance(adapter_build, Mapping)
            ):
                return False
            adapter_path = Path(str(adapter_ref["path"])).resolve()
            if not adapter_path.is_file() or adapter_ref.get("sha256") != _sha256_file(adapter_path):
                return False
            adapter_environment = {
                "HOME": str(child_path.parent / "home"),
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "UV_CACHE_DIR": str(child_path.parent / "uv-cache"),
                "UV_PROJECT": expected_working_directory,
                "UV_PROJECT_ENVIRONMENT": str(child_path.parent / "uv-environment"),
            }
            expected_build_command = [
                str(executable), "sync", "--locked", "--project",
                expected_working_directory, "--directory", expected_working_directory,
            ]
            build_stdout = child_path.parent / "adapter-build.stdout.bin"
            build_stderr = child_path.parent / "adapter-build.stderr.bin"
            if (
                set(adapter_build) != {
                    "command", "environment", "environment_sha256", "exit_code",
                    "stdout_sha256", "stderr_sha256",
                }
                or adapter_build.get("command") != expected_build_command
                or adapter_build.get("environment") != adapter_environment
                or adapter_build.get("environment_sha256") != _canonical_json_sha256(adapter_environment)
                or adapter_build.get("exit_code") != 0
                or not build_stdout.is_file()
                or not build_stderr.is_file()
                or adapter_build.get("stdout_sha256") != _sha256_file(build_stdout)
                or adapter_build.get("stderr_sha256") != _sha256_file(build_stderr)
            ):
                return False
            adapter_hash = _sha256_bytes(_TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8"))
            expected_adapter = [
                str(Path("/usr/bin/env").resolve()), "-i",
                *[f"{key}={value}" for key, value in sorted(adapter_environment.items())],
                str(executable), "run", "--locked", "--no-sync", "--project",
                expected_working_directory, "--directory", expected_working_directory,
                "python", "-c", _TOOFLASHY_PARITY_ADAPTER_SCRIPT,
                str(Path(str(input_payload["path"])).resolve()),
                str((child_path.parent / str(raw_output["path"])).resolve()),
                adapter_hash,
            ]
            return (
                executable.is_file()
                and _sha256_file(executable) == entry.get("binary_sha256")
                and Path(str(comparator_payload.get("binary", ""))).resolve() == executable
                and comparator_payload.get("binary_sha256") == entry.get("binary_sha256")
                and comparator_payload.get("working_directory") == expected_working_directory
                and launcher_identity.get("PWD") == expected_working_directory
                and launcher_identity.get("UV_PROJECT") == expected_working_directory
                and tool_command == expected_adapter
            )
        return (
            executable.is_file()
            and _sha256_file(executable) == entry.get("binary_sha256")
            and Path(str(comparator_payload.get("binary", ""))).resolve() == executable
            and comparator_payload.get("binary_sha256") == entry.get("binary_sha256")
            and comparator_payload.get("working_directory") == expected_working_directory
            and launcher_identity.get("PWD") == expected_working_directory
            and launcher_identity.get("UV_PROJECT") == expected_working_directory
            and tool_command == expected
        )
    if comparator == EA_IRIS_SOURCE_ADAPTER_ID:
        try:
            reopened, _, reopened_observation, reopened_path = _load_iris_source_adapter_run(
                child_path,
                expected_lane="DIRECT_DETECTOR",
            )
        except ExternalLeagueError:
            return False
        expected_thread_limit = protocol.get("threads", {}).get("limit")
        child_tool_command = child.get("tool_command")
        if (
            reopened_path != child_path.resolve()
            or reopened != dict(child)
            or not isinstance(expected_thread_limit, int)
            or isinstance(expected_thread_limit, bool)
            or not isinstance(child_tool_command, list)
            or len(child_tool_command) != 8
            or child_tool_command[-1] != str(expected_thread_limit)
            or reopened_observation.get("decoder_thread_control") != "VERIFIED"
            or reopened_observation.get("runtime_timing_eligible") is not True
            or tool_command != child_tool_command
            or not isinstance(input_payload, Mapping)
            or Path(str(input_payload.get("path", ""))).resolve() != canonical_input
            or input_payload.get("sha256") != _sha256_file(canonical_input)
            or launcher_identity != {
                "PWD": str(child_path.parent.resolve()),
                "UV_PROJECT": None,
            }
        ):
            return False
        build_ref = child.get("build_receipt")
        if not isinstance(build_ref, Mapping) or not isinstance(build_ref.get("path"), str):
            return False
        try:
            build, build_path = _load_iris_source_build_receipt(build_ref["path"])
        except ExternalLeagueError:
            return False
        binary = _iris_built_binary(build_path, build, "source_frame_adapter")
        return (
            build_ref.get("sha256") == _sha256_file(build_path)
            and Path(str(comparator_payload.get("binary", ""))).resolve() == binary
            and comparator_payload.get("binary_sha256") == _sha256_file(binary)
            and comparator_payload.get("working_directory") == str(child_path.parent.resolve())
        )
    if comparator == "FlashPatch":
        video = (conversion.parent / "canonical.ffv1.mkv").resolve()
        expected = [
            sys.executable,
            "-c",
            _FLASHPATCH_WORKER_SCRIPT,
            str(video),
            str(conversion),
            str(child_path.parent.resolve()),
        ]
        return (
            tool_command == expected
            and launcher_identity == {
                "PWD": str(Path(__file__).resolve().parents[2]),
                "UV_PROJECT": None,
            }
        )
    return False


def write_scheduled_runtime_repeat_receipt(
    comparator: str,
    run_receipts: Sequence[Path | str],
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    receipt_path: Path | str,
) -> dict[str, object]:
    """Assemble three already interleaved child runs without executing retries."""
    schema_by_comparator = {
        "FlashPatch": "flashpatch-l7-direct-detector-repeats-v1",
        KAYA_DIRECT_PARTICIPANT_ID: KAYA_FAIR_RUNTIME_REPEATS_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-repeats-v1",
    }
    child_schema_by_comparator = {
        "FlashPatch": "flashpatch-l7-direct-detector-run-v1",
        KAYA_DIRECT_PARTICIPANT_ID: KAYA_FAIR_RUNTIME_RUN_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-run-v1",
    }
    if comparator not in schema_by_comparator:
        raise ExternalLeagueError("scheduled repeat receipt comparator is unsupported")
    destination = Path(receipt_path).resolve()
    if destination.exists():
        raise FileExistsError(f"scheduled repeat receipt already exists: {destination}")
    frozen = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen is None:
        raise ExternalLeagueError("scheduled repeat receipt requires a runtime protocol")
    rows: list[dict[str, object]] = []
    for child_ref in run_receipts:
        child_path = Path(child_ref).resolve()
        try:
            child = json.loads(child_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({
                "repeat": None,
                "status": "INCONCLUSIVE",
                "receipt": str(child_path),
                "reason": f"child_receipt_unreadable:{exc.__class__.__name__}",
            })
            continue
        comparator_payload = child.get("comparator") if isinstance(child, Mapping) else None
        runtime = child.get("fair_runtime") if isinstance(child, Mapping) else None
        child_comparator = comparator_payload.get("name") if isinstance(comparator_payload, Mapping) else None
        if (
            not isinstance(child, Mapping)
            or child.get("schema") != child_schema_by_comparator[comparator]
            or child_comparator != comparator
            or not isinstance(runtime, Mapping)
        ):
            rows.append({
                "repeat": None,
                "status": "INCONCLUSIVE",
                "receipt": str(child_path),
                "receipt_sha256": _sha256_file(child_path),
                "reason": "child_receipt_identity_or_runtime_invalid",
            })
            continue
        observation = child.get("observation")
        if not isinstance(observation, Mapping):
            observation = child.get("parsed_observation")
        repeat_ordinal = runtime.get("scheduled_repeat_ordinal")
        rows.append({
            "repeat": repeat_ordinal,
            "status": child.get("status"),
            "receipt": str(child_path),
            "receipt_sha256": _sha256_file(child_path),
            "normalized_observation_sha256": (
                _canonical_json_sha256(observation)
                if isinstance(observation, Mapping)
                else None
            ),
            "fair_runtime": dict(runtime),
        })
    rows.sort(key=lambda row: row.get("repeat") if isinstance(row.get("repeat"), int) else 99)
    observation_hashes = {
        row.get("normalized_observation_sha256")
        for row in rows
        if row.get("status") == "PROCESS_VALID"
    }
    repeat_ordinals = [row.get("repeat") for row in rows]
    reproducible = (
        len(rows) == 3
        and repeat_ordinals == [1, 2, 3]
        and all(row.get("status") == "PROCESS_VALID" for row in rows)
        and len(observation_hashes) == 1
        and None not in observation_hashes
    )
    receipt = {
        "schema": schema_by_comparator[comparator],
        "repeats_required": 3,
        "comparator": comparator,
        "fair_runtime_protocol": frozen,
        "fair_runtime_protocol_sha256": _canonical_json_sha256(frozen),
        "runs": rows,
        "status": "PROCESS_REPRODUCIBLE" if reproducible else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": [
            *([] if reproducible else ["scheduled_run_set_inconclusive"]),
            *(
                ["independent_execution_witness_missing"]
                if comparator == KAYA_DIRECT_PARTICIPANT_ID
                else []
            ),
            "independent_gold_receipt_missing",
            "frozen_public_case_ledger_missing",
        ],
    }
    if comparator == KAYA_DIRECT_PARTICIPANT_ID:
        receipt.update({
            "claim_status": "NOT_SCOREABLE",
            "comparison_eligible": False,
            "external_claim_authorized": False,
        })
    _write_json(destination, receipt)
    return {**receipt, "receipt": str(destination)}


def _portable_schedule_binding(binding: Mapping[str, object]) -> dict[str, object]:
    """Project the host-local schedule binding down to its portable identity."""
    fields = (
        "schedule_sha256",
        "slot",
        "round",
        "position",
        "comparator",
        "repeat_ordinal",
    )
    return {field: binding.get(field) for field in fields}


def _verify_external_slot_child_joins(
    *,
    schedule: Mapping[str, object],
    schedule_sha256: str,
    local_children: Mapping[tuple[str, int], Mapping[str, object]],
    external_verification: Mapping[str, object],
) -> tuple[bool, list[dict[str, object]], list[str]]:
    """Join each verified external result to one locally re-parsed child receipt."""
    failures: list[str] = []
    joined: list[dict[str, object]] = []
    expected_schedule_slots = schedule.get("slots")
    verified_slots = external_verification.get("verified_slots")
    if (
        not isinstance(expected_schedule_slots, list)
        or len(expected_schedule_slots) != 9
        or not isinstance(verified_slots, list)
        or len(verified_slots) != 9
        or len(local_children) != 9
    ):
        return False, [], ["external_slot_population_invalid"]
    expected_by_slot = {
        row.get("slot"): row
        for row in expected_schedule_slots
        if isinstance(row, Mapping)
    }
    if set(expected_by_slot) != set(range(1, 10)):
        return False, [], ["external_slot_population_invalid"]
    observed_slots: set[int] = set()
    observed_children: set[str] = set()
    expected_slot_fields = {
        "slot",
        "round",
        "position",
        "comparator",
        "repeat_ordinal",
        "result",
    }
    result_fields = {
        "schema",
        "status",
        "slot",
        "comparator",
        "repeat_ordinal",
        "child_receipt_sha256",
        "ffv1_input_sha256",
        "schedule_binding",
        "parser_observation",
    }
    for external_slot in verified_slots:
        if not isinstance(external_slot, Mapping) or set(external_slot) != expected_slot_fields:
            failures.append("external_slot_projection_invalid")
            continue
        slot = external_slot.get("slot")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot in observed_slots:
            failures.append("external_slot_projection_invalid")
            continue
        observed_slots.add(slot)
        scheduled = expected_by_slot.get(slot)
        if not isinstance(scheduled, Mapping) or any(
            external_slot.get(field) != scheduled.get(field)
            for field in ("slot", "round", "position", "comparator", "repeat_ordinal")
        ):
            failures.append(f"external_slot_schedule_mismatch:{slot}")
            continue
        comparator = external_slot.get("comparator")
        ordinal = external_slot.get("repeat_ordinal")
        local = (
            local_children.get((str(comparator), int(ordinal)))
            if isinstance(ordinal, int) and not isinstance(ordinal, bool)
            else None
        )
        artifact = external_slot.get("result")
        if (
            not isinstance(local, Mapping)
            or not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256", "size"}
        ):
            failures.append(f"external_slot_child_join_missing:{comparator}:{ordinal}")
            continue
        result_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            if (
                result_path.is_symlink()
                or not result_path.is_file()
                or artifact.get("sha256") != _sha256_file(result_path)
                or artifact.get("size") != result_path.stat().st_size
            ):
                raise ExternalLeagueError("external result artifact drifted")
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ExternalLeagueError):
            failures.append(f"external_slot_result_invalid:{slot}")
            continue
        if not isinstance(result_payload, Mapping) or set(result_payload) != result_fields:
            failures.append(f"external_slot_result_invalid:{slot}")
            continue
        local_binding = local.get("schedule_binding")
        expected_result = {
            "schema": EXTERNAL_SLOT_CHILD_JOIN_SCHEMA,
            "status": "PROCESS_VALID",
            "slot": slot,
            "comparator": comparator,
            "repeat_ordinal": ordinal,
            "child_receipt_sha256": local.get("receipt_sha256"),
            "ffv1_input_sha256": local.get("input_sha256"),
            "schedule_binding": (
                _portable_schedule_binding(local_binding)
                if isinstance(local_binding, Mapping)
                else None
            ),
            "parser_observation": local.get("parser_observation"),
        }
        portable_binding = expected_result["schedule_binding"]
        if (
            not isinstance(portable_binding, Mapping)
            or portable_binding.get("schedule_sha256") != schedule_sha256
            or dict(result_payload) != expected_result
        ):
            failures.append(f"external_slot_child_join_mismatch:{comparator}:{ordinal}")
            continue
        child_sha = result_payload.get("child_receipt_sha256")
        if not isinstance(child_sha, str) or child_sha in observed_children:
            failures.append(f"external_slot_child_not_one_to_one:{comparator}:{ordinal}")
            continue
        observed_children.add(child_sha)
        joined.append(
            {
                "slot": slot,
                "comparator": comparator,
                "repeat_ordinal": ordinal,
                "external_result_sha256": artifact["sha256"],
                "child_receipt_sha256": child_sha,
            }
        )
    expected_children = {
        str(row.get("receipt_sha256")) for row in local_children.values()
    }
    if (
        observed_slots != set(range(1, 10))
        or len(joined) != 9
        or observed_children != expected_children
        or len(expected_children) != 9
    ):
        failures.append("external_slot_child_join_incomplete")
    return not failures, joined, failures


def verify_fair_runtime_receipts(
    repeat_receipts: Sequence[Path | str],
    *,
    schedule_receipt: Path | str | None = None,
    external_host_witness: Mapping[str, Path | str] | None = None,
) -> dict[str, object]:
    """Verify equal-budget runs and, when supplied, their pre-frozen order."""
    failures: list[str] = []
    receipt_rows: list[dict[str, object]] = []
    protocol_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    effective_environment_hashes: dict[str, set[str]] = {}
    input_hashes: set[str] = set()
    comparators: list[str] = []
    runtime_rows: list[dict[str, object]] = []
    local_children: dict[tuple[str, int], dict[str, object]] = {}
    direct_repeat_schema_by_comparator = {
        "FlashPatch": "flashpatch-l7-direct-detector-repeats-v1",
        KAYA_DIRECT_PARTICIPANT_ID: KAYA_FAIR_RUNTIME_REPEATS_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-repeats-v1",
    }
    schedule: dict[str, object] | None = None
    schedule_path: Path | None = None
    schedule_artifact_sha256: str | None = None
    schedule_sha256: str | None = None
    schedule_stat: dict[str, int] | None = None
    external_witness_verification: dict[str, object] | None = None
    if schedule_receipt is not None:
        if not isinstance(schedule_receipt, (str, os.PathLike)):
            failures.append("fair_runtime_schedule_invalid")
        else:
            schedule_path = Path(schedule_receipt).resolve()
            try:
                schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
                schedule = _validate_frozen_runtime_schedule(schedule_payload)
                schedule_artifact_sha256 = _sha256_file(schedule_path)
                schedule_sha256 = _canonical_json_sha256(schedule)
                observed_stat = schedule_path.stat()
                schedule_stat = {
                    "device": observed_stat.st_dev,
                    "inode": observed_stat.st_ino,
                    "size": observed_stat.st_size,
                    "mtime_ns": observed_stat.st_mtime_ns,
                    "ctime_ns": observed_stat.st_ctime_ns,
                }
            except (OSError, json.JSONDecodeError, ExternalLeagueError):
                failures.append("fair_runtime_schedule_invalid")
    receipt_collection_valid = (
        isinstance(repeat_receipts, Sequence)
        and not isinstance(repeat_receipts, (str, bytes))
    )
    if (
        not receipt_collection_valid
        or len(repeat_receipts) < 2
    ):
        failures.append("runtime_bundle_requires_at_least_two_comparators")
    receipt_refs = repeat_receipts if receipt_collection_valid else ()
    for receipt_ref in receipt_refs:
        if not isinstance(receipt_ref, (str, os.PathLike)):
            failures.append("repeat_receipt_reference_invalid")
            continue
        path = Path(receipt_ref).resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"repeat_receipt_unreadable:{path}")
            continue
        if not isinstance(payload, Mapping):
            failures.append(f"repeat_receipt_schema_invalid:{path}")
            continue
        comparator = payload.get("comparator")
        if not isinstance(comparator, str) or not comparator:
            failures.append(f"comparator_identity_missing:{path}")
            continue
        if (
            comparator not in DIRECT_DETECTOR_POPULATION
            or payload.get("schema") != direct_repeat_schema_by_comparator.get(comparator)
        ):
            failures.append(f"repeat_receipt_schema_invalid:{path}")
            continue
        comparators.append(comparator)
        effective_environment_hashes.setdefault(comparator, set())
        if comparator == KAYA_DIRECT_PARTICIPANT_ID and (
            payload.get("claim_status") != "NOT_SCOREABLE"
            or payload.get("scoreable") is not False
            or payload.get("comparison_eligible") is not False
            or payload.get("external_claim_authorized") is not False
        ):
            failures.append(f"claim_boundary_invalid:{comparator}")
        if payload.get("repeats_required") != 3:
            failures.append(f"scheduled_repeat_budget_invalid:{comparator}")
        if payload.get("status") != "PROCESS_REPRODUCIBLE":
            failures.append(f"repeat_receipt_inconclusive:{comparator}")
        try:
            frozen = _validate_frozen_runtime_protocol(payload.get("fair_runtime_protocol"))
        except ExternalLeagueError:
            failures.append(f"runtime_protocol_invalid:{comparator}")
            continue
        protocol_sha256 = _canonical_json_sha256(frozen)
        if payload.get("fair_runtime_protocol_sha256") != protocol_sha256:
            failures.append(f"runtime_protocol_hash_mismatch:{comparator}")
        protocol_hashes.add(protocol_sha256)
        environment_sha256 = _runtime_environment_sha256(frozen)
        environment_hashes.add(environment_sha256)
        budget = frozen["budget"]
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 3:
            failures.append(f"scheduled_repeat_count_invalid:{comparator}")
            continue
        ordinals = [run.get("repeat") for run in runs if isinstance(run, Mapping)]
        if (
            len(ordinals) != 3
            or any(isinstance(ordinal, bool) or not isinstance(ordinal, int) for ordinal in ordinals)
            or sorted(ordinals) != [1, 2, 3]
            or len(set(ordinals)) != 3
        ):
            failures.append(f"scheduled_repeat_ordinals_invalid:{comparator}")
            continue
        observation_hashes: set[str] = set()
        normalizers: set[str] = set()
        for run in runs:
            if not isinstance(run, Mapping):
                failures.append(f"repeat_row_invalid:{comparator}")
                continue
            ordinal = run["repeat"]
            if run.get("status") != "PROCESS_VALID":
                failures.append(f"scheduled_run_inconclusive:{comparator}:{ordinal}")
            runtime = run.get("fair_runtime")
            if not isinstance(runtime, Mapping) or runtime.get("schema") != FAIR_RUNTIME_RUN_SCHEMA:
                failures.append(f"fair_runtime_run_missing:{comparator}:{ordinal}")
                continue
            expected_runtime_fields = {
                "schema",
                "protocol_sha256",
                "measurement_boundary",
                "environment_policy_sha256",
                "observed_environment",
                "timeout_seconds",
                "scheduled_repeat_ordinal",
                "schedule_binding",
                "attempt_ordinal",
                "retry_count",
                "retry_policy",
                "started_monotonic_ns",
                "finished_monotonic_ns",
                "wall_time_ns",
                "timed_out",
                "input_identity_sha256",
                "normalized_terminal_observation",
            }
            if set(runtime) != expected_runtime_fields:
                failures.append(f"fair_runtime_run_fields_invalid:{comparator}:{ordinal}")
                continue
            if runtime.get("protocol_sha256") != protocol_sha256:
                failures.append(f"run_protocol_mismatch:{comparator}:{ordinal}")
            if runtime.get("measurement_boundary") != FAIR_RUNTIME_BOUNDARY:
                failures.append(f"measurement_boundary_mismatch:{comparator}:{ordinal}")
            if runtime.get("environment_policy_sha256") != environment_sha256:
                failures.append(f"environment_policy_mismatch:{comparator}:{ordinal}")
            if runtime.get("timeout_seconds") != budget["timeout_seconds"]:
                failures.append(f"timeout_budget_mismatch:{comparator}:{ordinal}")
            if runtime.get("scheduled_repeat_ordinal") != ordinal:
                failures.append(f"scheduled_repeat_ordinal_mismatch:{comparator}:{ordinal}")
            if (
                runtime.get("attempt_ordinal") != 1
                or runtime.get("retry_count") != 0
                or runtime.get("retry_policy") != "NO_RETRY"
            ):
                failures.append(f"retry_detected:{comparator}:{ordinal}")
            started = runtime.get("started_monotonic_ns")
            finished = runtime.get("finished_monotonic_ns")
            wall_time_ns = runtime.get("wall_time_ns")
            if (
                isinstance(started, bool)
                or not isinstance(started, int)
                or started < 0
                or isinstance(finished, bool)
                or not isinstance(finished, int)
                or finished <= started
                or isinstance(wall_time_ns, bool)
                or not isinstance(wall_time_ns, int)
                or wall_time_ns <= 0
                or finished - started != wall_time_ns
            ):
                failures.append(f"monotonic_interval_invalid:{comparator}:{ordinal}")
            if (
                isinstance(wall_time_ns, int)
                and not isinstance(wall_time_ns, bool)
                and wall_time_ns > int(budget["timeout_seconds"]) * 1_000_000_000
            ):
                failures.append(f"wall_time_exceeds_budget:{comparator}:{ordinal}")
            if runtime.get("timed_out") is not False:
                failures.append(f"scheduled_run_timed_out:{comparator}:{ordinal}")
            input_sha256 = runtime.get("input_identity_sha256")
            binding = runtime.get("schedule_binding")
            if not isinstance(input_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
                failures.append(f"input_identity_invalid:{comparator}:{ordinal}")
            else:
                input_hashes.add(input_sha256)
                if not _observed_environment_matches_protocol(
                    runtime.get("observed_environment"),
                    frozen,
                    input_sha256,
                    binding if isinstance(binding, Mapping) else None,
                ):
                    failures.append(f"observed_environment_invalid:{comparator}:{ordinal}")
                else:
                    child_probe = runtime["observed_environment"]["child_probe"]
                    effective_environment_hashes[comparator].add(
                        str(child_probe["effective_environment_sha256"])
                    )
            if schedule is not None:
                if not isinstance(binding, Mapping):
                    failures.append(f"schedule_binding_missing:{comparator}:{ordinal}")
            terminal = runtime.get("normalized_terminal_observation")
            if (
                not isinstance(terminal, Mapping)
                or terminal.get("schema") != "flashpatch-l7-normalized-terminal-observation-v1"
                or not isinstance(terminal.get("normalizer"), str)
                or not terminal.get("normalizer")
                or not isinstance(terminal.get("implementation"), Mapping)
                or not isinstance(terminal.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(terminal.get("sha256"))) is None
            ):
                failures.append(f"normalized_terminal_observation_missing:{comparator}:{ordinal}")
                continue
            observation_hashes.add(str(terminal["sha256"]))
            normalizers.add(str(terminal["normalizer"]))
            implementation = terminal["implementation"]
            implementation_path = Path(str(implementation.get("path", ""))).resolve()
            if (
                set(implementation) != {"path", "sha256"}
                or not implementation_path.is_file()
                or implementation.get("sha256") != _sha256_file(implementation_path)
            ):
                failures.append(f"normalizer_implementation_invalid:{comparator}:{ordinal}")
            try:
                child_path = Path(str(run.get("receipt"))).resolve()
                if not child_path.is_file() or run.get("receipt_sha256") != _sha256_file(child_path):
                    raise ExternalLeagueError("child receipt hash mismatch")
                child = json.loads(child_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ExternalLeagueError):
                failures.append(f"child_run_receipt_invalid:{comparator}:{ordinal}")
                continue
            if not isinstance(child, Mapping) or child.get("fair_runtime") != runtime:
                failures.append(f"child_run_runtime_mismatch:{comparator}:{ordinal}")
                continue
            if not _child_probe_and_command_are_bound(child_path, child, runtime, frozen, comparator):
                failures.append(f"child_runtime_probe_or_command_unbound:{comparator}:{ordinal}")
                continue
            child_probe = runtime["observed_environment"]["child_probe"]
            child_timing = child_probe.get("child_timing")
            child_timing_values = (
                []
                if not isinstance(child_timing, Mapping)
                else [
                    child_timing.get("probe_started_monotonic_ns"),
                    child_timing.get("tool_started_monotonic_ns"),
                    child_timing.get("tool_finished_monotonic_ns"),
                ]
            )
            if (
                not isinstance(child_timing, Mapping)
                or not isinstance(started, int)
                or not isinstance(finished, int)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in child_timing_values
                )
                or not (
                    started
                    <= child_timing.get("probe_started_monotonic_ns", -1)
                    <= child_timing.get("tool_started_monotonic_ns", -1)
                    < child_timing.get("tool_finished_monotonic_ns", -1)
                    <= finished
                )
            ):
                failures.append(f"child_parent_timing_mismatch:{comparator}:{ordinal}")
                continue
            if schedule is not None and isinstance(binding, Mapping):
                runtime_rows.append({
                    "comparator": comparator,
                    "ordinal": ordinal,
                    "binding": dict(binding),
                    "started": child_timing["tool_started_monotonic_ns"],
                    "finished": child_timing["tool_finished_monotonic_ns"],
                })
            try:
                observation = _reopen_child_normalized_observation(
                    child_path,
                    child,
                    comparator,
                    str(input_sha256),
                )
            except ExternalLeagueError:
                failures.append(f"raw_evidence_reparse_failed:{comparator}:{ordinal}")
                continue
            child_observation = child.get("observation")
            if not isinstance(child_observation, Mapping):
                child_observation = child.get("parsed_observation")
            if (
                not isinstance(child_observation, Mapping)
                or dict(child_observation) != observation
                or _canonical_json_sha256(observation) != terminal["sha256"]
            ):
                failures.append(f"normalized_terminal_observation_unbound:{comparator}:{ordinal}")
            else:
                child_key = (comparator, ordinal)
                if child_key in local_children:
                    failures.append(f"duplicate_child_run:{comparator}:{ordinal}")
                else:
                    local_children[child_key] = {
                        "receipt": str(child_path),
                        "receipt_sha256": _sha256_file(child_path),
                        "input_sha256": input_sha256,
                        "schedule_binding": dict(binding) if isinstance(binding, Mapping) else None,
                        "parser_observation": observation,
                    }
        if len(observation_hashes) != 1:
            failures.append(f"normalized_repeat_disagreement:{comparator}")
        if len(normalizers) != 1:
            failures.append(f"normalizer_identity_disagreement:{comparator}")
        receipt_rows.append({
            "comparator": comparator,
            "receipt": str(path),
            "sha256": _sha256_file(path),
            "protocol_sha256": protocol_sha256,
        })
    if len(comparators) != len(set(comparators)):
        failures.append("duplicate_comparator_receipt")
    if len(protocol_hashes) != 1:
        failures.append("unequal_runtime_protocol")
    if len(environment_hashes) != 1:
        failures.append("unequal_environment_policy")
    for comparator, hashes in effective_environment_hashes.items():
        if len(hashes) != 1:
            failures.append(f"effective_environment_drift_between_repeats:{comparator}")
    all_effective_hashes = {
        value
        for hashes in effective_environment_hashes.values()
        for value in hashes
    }
    if len(all_effective_hashes) != 1:
        failures.append("unequal_effective_environment")
    if len(input_hashes) != 1:
        failures.append("unequal_canonical_input")
    if schedule is not None and schedule_path is not None:
        if schedule["participants"] != sorted(comparators):
            failures.append("schedule_participant_population_mismatch")
        if len(protocol_hashes) != 1 or schedule["protocol_sha256"] not in protocol_hashes:
            failures.append("schedule_protocol_binding_mismatch")
        if len(input_hashes) != 1 or schedule["input_sha256"] not in input_hashes:
            failures.append("schedule_input_binding_mismatch")
        expected_slots = {entry["slot"]: entry for entry in schedule["slots"]}
        actual_slots: dict[int, dict[str, object]] = {}
        for row in runtime_rows:
            binding = row["binding"]
            slot = binding.get("slot")
            if isinstance(slot, bool) or not isinstance(slot, int) or slot in actual_slots:
                failures.append("schedule_slot_duplicate_or_invalid")
                continue
            expected = expected_slots.get(slot)
            expected_binding = None if expected is None else {
                "path": str(schedule_path),
                "artifact_sha256": schedule_artifact_sha256,
                "schedule_sha256": schedule_sha256,
                "stat": schedule_stat,
                **expected,
            }
            if (
                expected_binding is None
                or binding != expected_binding
                or binding.get("comparator") != row["comparator"]
                or binding.get("repeat_ordinal") != row["ordinal"]
            ):
                failures.append(f"schedule_assignment_mismatch:{row['comparator']}:{row['ordinal']}")
                continue
            actual_slots[slot] = row
        if set(actual_slots) != set(expected_slots):
            failures.append("schedule_slot_set_incomplete")
        ordered_rows = [actual_slots[slot] for slot in sorted(actual_slots)]
        for previous, current in zip(ordered_rows, ordered_rows[1:]):
            if (
                not isinstance(previous.get("finished"), int)
                or not isinstance(current.get("started"), int)
                or current["started"] < previous["finished"]
            ):
                failures.append("schedule_execution_order_or_isolation_invalid")
                break
    if external_host_witness is not None:
        if (
            not isinstance(external_host_witness, Mapping)
            or set(external_host_witness) != {"request", "receipt"}
            or not all(
                isinstance(external_host_witness.get(field), (str, os.PathLike))
                for field in ("request", "receipt")
            )
            or schedule is None
            or schedule_sha256 is None
            or len(protocol_hashes) != 1
            or len(input_hashes) != 1
        ):
            failures.append("external_host_witness_invalid")
        else:
            external_witness_verification = verify_external_host_witness(
                external_host_witness["request"],
                external_host_witness["receipt"],
                expected_protocol_sha256=next(iter(protocol_hashes)),
                expected_schedule_sha256=schedule_sha256,
                expected_input_sha256=next(iter(input_hashes)),
                local_host_identity=capture_external_host_identity(),
            )
            if external_witness_verification.get("witness_verified") is not True:
                failures.append("external_host_witness_invalid")
    external_witness_verified = (
        external_witness_verification is not None
        and external_witness_verification.get("witness_verified") is True
        and external_witness_verification.get("schema")
        == EXTERNAL_HOST_VERIFICATION_SCHEMA_V2
    )
    legacy_external_witness_verified = (
        external_witness_verification is not None
        and external_witness_verification.get("witness_verified") is True
        and not external_witness_verified
    )
    full_population = (
        schedule is not None
        and schedule.get("participants") == sorted(DIRECT_DETECTOR_POPULATION)
        and isinstance(schedule.get("slots"), list)
        and len(schedule["slots"]) == 9
        and len(comparators) == len(DIRECT_DETECTOR_POPULATION)
        and set(comparators) == set(DIRECT_DETECTOR_POPULATION)
    )
    external_slot_child_joins: list[dict[str, object]] = []
    external_slot_join_verified = False
    if (
        external_witness_verified
        and full_population
        and schedule is not None
        and schedule_sha256 is not None
        and external_witness_verification is not None
        and not failures
    ):
        (
            external_slot_join_verified,
            external_slot_child_joins,
            join_failures,
        ) = _verify_external_slot_child_joins(
            schedule=schedule,
            schedule_sha256=schedule_sha256,
            local_children=local_children,
            external_verification=external_witness_verification,
        )
        failures.extend(join_failures)
    receipts_verified = not failures
    schedule_environment_verified = receipts_verified and schedule is not None
    fair_runtime_verified = (
        receipts_verified
        and external_witness_verified
        and full_population
        and external_slot_join_verified
    )
    status = (
        "INCONCLUSIVE"
        if failures
        else "NOT_VERIFIED"
    )
    effective_hashes = {
        comparator: sorted(hashes)
        for comparator, hashes in effective_environment_hashes.items()
    }
    comparison_blockers: list[str] = []
    if schedule is None:
        comparison_blockers.append("balanced_interleaved_schedule_missing")
    elif external_host_witness is None and schedule_environment_verified:
        comparison_blockers.append("independent_execution_witness_missing")
    elif legacy_external_witness_verified:
        comparison_blockers.append("external_host_witness_v2_required")
    elif external_host_witness is not None and not external_witness_verified:
        comparison_blockers.append("external_host_witness_invalid")
    elif external_witness_verified and full_population and not external_slot_join_verified:
        comparison_blockers.append("external_slot_child_join_invalid")
    elif external_witness_verified and not fair_runtime_verified:
        comparison_blockers.append("fair_population_receipt_conditions_unproven")
    if "unequal_effective_environment" in failures:
        comparison_blockers.append("unequal_effective_environment")
    return {
        "schema": FAIR_RUNTIME_BUNDLE_SCHEMA,
        "status": status,
        "receipts_verified": receipts_verified,
        "schedule_environment_verified": schedule_environment_verified,
        "independent_execution_witness_verified": external_witness_verified,
        "fair_runtime_verified": fair_runtime_verified,
        "runtime_comparison_ready": False,
        "claim_status": "NOT_SCOREABLE",
        "comparison_eligible": False,
        "schedule": (
            {
                "path": str(schedule_path),
                "artifact_sha256": schedule_artifact_sha256,
                "schedule_sha256": schedule_sha256,
            }
            if schedule is not None and schedule_path is not None
            else None
        ),
        "receipts": receipt_rows,
        "failures": failures,
        "effective_environment_sha256": effective_hashes,
        "external_host_witness": external_witness_verification,
        "external_slot_child_joins": external_slot_child_joins,
        "runtime_comparison_blockers": comparison_blockers,
        "scoreable": False,
        "scoreable_blockers": [
            *comparison_blockers,
            "independent_gold_receipt_missing",
            "frozen_public_case_ledger_missing",
            "receipt_bound_score_verifier_missing",
        ],
    }


def _audit_kaya_source_checkout(checkout: Path | str) -> dict[str, object]:
    """Verify the exact unmodified Kaya source used by the UNSCORED prototype."""
    root = Path(checkout).resolve()
    if not root.is_dir():
        raise ExternalLeagueError("Kaya source checkout is missing")
    for relative, expected in KAYA_REQUIRED_SOURCE_HASHES.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError("Kaya source path escapes checkout") from exc
        if not path.is_file() or _sha256_file(path) != expected:
            raise ExternalLeagueError(f"Kaya pinned source hash drifted: {relative}")
    commands = {
        "revision": ["/usr/bin/git", "rev-parse", "HEAD"],
        "tree": ["/usr/bin/git", "rev-parse", "HEAD^{tree}"],
        "status": ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"],
        "remote": ["/usr/bin/git", "remote", "get-url", "origin"],
    }
    observations: dict[str, str] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        if completed.returncode != 0 or completed.stderr:
            raise ExternalLeagueError(f"Kaya source provenance command failed: {name}")
        observations[name] = completed.stdout.strip()
    if observations["revision"] != KAYA_SOURCE_REVISION or observations["tree"] != KAYA_SOURCE_TREE:
        raise ExternalLeagueError("Kaya source revision or tree differs from the frozen prototype")
    if observations["status"]:
        raise ExternalLeagueError("Kaya source checkout is not clean")
    if observations["remote"].removesuffix(".git") != KAYA_REPOSITORY_URL:
        raise ExternalLeagueError("Kaya source remote differs from the frozen repository")
    return {
        "repository_url": KAYA_REPOSITORY_URL,
        "revision": KAYA_SOURCE_REVISION,
        "tree": KAYA_SOURCE_TREE,
        "license": "BSD-3-Clause",
        "source_hashes": dict(KAYA_REQUIRED_SOURCE_HASHES),
    }


def _kaya_alternating_frames(
    count: int,
    *,
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    hold: int,
    height: int = 72,
    width: int = 96,
) -> np.ndarray:
    if count <= 0 or hold <= 0:
        raise ExternalLeagueError("Kaya fixture frame count and hold must be positive")
    frames = np.empty((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        frames[index] = first if (index // hold) % 2 == 0 else second
    return frames


def materialize_kaya_conformance_fixtures(output_root: Path | str) -> dict[str, dict[str, object]]:
    """Create controlled, non-scoring renderer/FFV1 conformance fixtures."""
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Kaya fixture root already exists: {root}")
    root.mkdir(parents=True)
    renderer_root = root / "renderer"
    conversion_root = root / "conversion"
    renderer_root.mkdir()
    conversion_root.mkdir()
    fixtures: dict[str, np.ndarray] = {
        "safe": np.full((61, 72, 96, 3), 96, dtype=np.uint8),
        "rgb-channel-trap": _kaya_alternating_frames(
            72, first=(255, 0, 0), second=(200, 0, 0), hold=4
        ),
        "flash-threshold": _kaya_alternating_frames(
            90, first=(0, 0, 0), second=(255, 255, 255), hold=4
        ),
        "history-59": _kaya_alternating_frames(
            59, first=(0, 0, 0), second=(255, 255, 255), hold=4
        ),
        "history-60": _kaya_alternating_frames(
            60, first=(0, 0, 0), second=(255, 255, 255), hold=4
        ),
        "history-61": _kaya_alternating_frames(
            61, first=(0, 0, 0), second=(255, 255, 255), hold=4
        ),
        "letterbox": _kaya_alternating_frames(
            61, first=(8, 8, 8), second=(248, 248, 248), hold=5,
            height=36, width=180,
        ),
        "state-reuse": _kaya_alternating_frames(
            61, first=(255, 0, 0), second=(255, 255, 255), hold=3
        ),
    }
    if tuple(fixtures) != KAYA_REQUIRED_FIXTURE_IDS:
        raise ExternalLeagueError("Kaya conformance fixture population drifted")
    rows: dict[str, dict[str, object]] = {}
    for fixture_id, frames in fixtures.items():
        source = renderer_root / f"{fixture_id}.npz"
        np.savez(source, frames=frames, timestamps=np.arange(len(frames), dtype=np.float64) / 60.0)
        conversion = materialize_cfr_ffv1(source, conversion_root / fixture_id, fps=60)
        conversion_path = Path(str(conversion["receipt"])).resolve()
        video = conversion_path.parent / "canonical.ffv1.mkv"
        rows[fixture_id] = {
            "fixture_id": fixture_id,
            "classification": "CONTROLLED_CONFORMANCE_NOT_NATURAL_NOT_SCORING",
            "renderer_source": str(source),
            "renderer_source_sha256": _sha256_file(source),
            "video": str(video),
            "video_sha256": _sha256_file(video),
            "conversion_receipt": str(conversion_path),
            "conversion_receipt_sha256": _sha256_file(conversion_path),
            "frame_count": len(frames),
            "shape": list(frames.shape),
            "fps": 60,
        }
    manifest = {
        "schema": "flashpatch-l7-kaya-conformance-fixtures-v1",
        "classification": "CONTROLLED_CONFORMANCE_NOT_NATURAL_NOT_SCORING",
        "fixture_ids": list(KAYA_REQUIRED_FIXTURE_IDS),
        "fixtures": rows,
    }
    _write_json(root / "fixtures-manifest.json", manifest)
    return {key: dict(value) for key, value in rows.items()}


def _materialize_kaya_direct_input(
    fixture_id: str,
    video: Path,
    conversion: Path,
    output_root: Path,
) -> dict[str, object]:
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    source = Path(str(contract["renderer_source"]["path"])).resolve()
    try:
        with np.load(source) as archive:
            frames = np.asarray(archive["frames"])
    except (KeyError, OSError, ValueError) as exc:
        raise ExternalLeagueError("Kaya renderer RGB source cannot be reopened") from exc
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ExternalLeagueError("Kaya renderer RGB source is not uint8 RGB")
    output_root.mkdir(parents=True)
    frames_path = output_root / "direct-rgb.npy"
    np.save(frames_path, frames, allow_pickle=False)
    ledger = [
        {
            "index": index,
            "cfr_timestamp_us": round(index * 1_000_000 / 60),
            "shape": list(frames[index].shape),
            "pixel_format": "rgb24",
            "rgb_sha256": _sha256_bytes(np.ascontiguousarray(frames[index]).tobytes()),
        }
        for index in range(len(frames))
    ]
    manifest = {
        "schema": KAYA_DIRECT_INPUT_SCHEMA,
        "fixture_id": fixture_id,
        "fps": 60,
        "frame_count": len(frames),
        "shape": list(frames.shape),
        "dtype": "uint8",
        "pixel_format": "rgb24",
        "frames": str(frames_path),
        "frames_file_sha256": _sha256_file(frames_path),
        "raw_rgb_sha256": _sha256_bytes(np.ascontiguousarray(frames).tobytes()),
        "ledger": ledger,
        "ledger_sha256": _canonical_json_sha256(ledger),
        "canonical_video": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
    }
    manifest_path = output_root / "direct-input.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path)}


def _load_kaya_direct_input_manifest(
    manifest_path: Path | str,
    *,
    expected_video: Path | str | None = None,
    expected_conversion: Path | str | None = None,
) -> dict[str, object]:
    path = Path(manifest_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("Kaya direct input manifest is unreadable") from exc
    fields = {
        "schema", "fixture_id", "fps", "frame_count", "shape", "dtype",
        "pixel_format", "frames", "frames_file_sha256", "raw_rgb_sha256",
        "ledger", "ledger_sha256", "canonical_video", "conversion_receipt",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ExternalLeagueError("Kaya direct input manifest fields are invalid")
    if (
        payload.get("schema") != KAYA_DIRECT_INPUT_SCHEMA
        or isinstance(payload.get("fps"), bool)
        or not isinstance(payload.get("fps"), int)
        or payload.get("fps") != 60
        or payload.get("dtype") != "uint8"
        or payload.get("pixel_format") != "rgb24"
    ):
        raise ExternalLeagueError("Kaya direct input must be exact 60 CFR uint8 RGB")
    frames_path = Path(str(payload.get("frames", ""))).resolve()
    if not frames_path.is_file() or payload.get("frames_file_sha256") != _sha256_file(frames_path):
        raise ExternalLeagueError("Kaya direct RGB artifact hash mismatches")
    try:
        frames = np.load(frames_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ExternalLeagueError("Kaya direct RGB artifact is unreadable") from exc
    frame_count = payload.get("frame_count")
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or len(frames) != frame_count
        or list(frames.shape) != payload.get("shape")
        or payload.get("raw_rgb_sha256") != _sha256_bytes(np.ascontiguousarray(frames).tobytes())
    ):
        raise ExternalLeagueError("Kaya direct RGB array contract is invalid")
    ledger = payload.get("ledger")
    if not isinstance(ledger, list) or len(ledger) != frame_count or payload.get("ledger_sha256") != _canonical_json_sha256(ledger):
        raise ExternalLeagueError("Kaya direct RGB ledger is invalid")
    observed_indices: set[int] = set()
    for index, row in enumerate(ledger):
        expected = {
            "index": index,
            "cfr_timestamp_us": round(index * 1_000_000 / 60),
            "shape": list(frames[index].shape),
            "pixel_format": "rgb24",
            "rgb_sha256": _sha256_bytes(np.ascontiguousarray(frames[index]).tobytes()),
        }
        if not isinstance(row, Mapping) or dict(row) != expected or row.get("index") in observed_indices:
            raise ExternalLeagueError("Kaya direct RGB ledger has a missing, duplicate, or malformed row")
        observed_indices.add(index)
    for label, expected_value in (
        ("canonical_video", expected_video),
        ("conversion_receipt", expected_conversion),
    ):
        reference = payload.get(label)
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ExternalLeagueError(f"Kaya direct input {label} binding is invalid")
        reference_path = Path(str(reference.get("path", ""))).resolve()
        if not reference_path.is_file() or reference.get("sha256") != _sha256_file(reference_path):
            raise ExternalLeagueError(f"Kaya direct input {label} hash mismatches")
        if expected_value is not None and reference_path != Path(expected_value).resolve():
            raise ExternalLeagueError(f"Kaya direct input {label} path mismatches")
    return dict(payload)


def _kaya_child_environment(environment_root: Path, home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": f"{environment_root / 'bin'}:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "QT_QPA_PLATFORM": "offscreen",
    }


def _kaya_runtime_base_evidence(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise ExternalLeagueError("Kaya interpreter base is unavailable")
    rows: list[dict[str, object]] = []
    content_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise ExternalLeagueError("Kaya interpreter base contains an absolute symlink")
            try:
                path.resolve(strict=False).relative_to(root)
            except ValueError as exc:
                raise ExternalLeagueError("Kaya interpreter base symlink escapes its root") from exc
            rows.append({"path": relative, "type": "symlink", "target": target})
        elif path.is_file():
            size = path.stat().st_size
            content_bytes += size
            rows.append({
                "path": relative,
                "type": "file",
                "bytes": size,
                "sha256": _sha256_file(path),
            })
        elif not path.is_dir():
            raise ExternalLeagueError("Kaya interpreter base contains a non-file artifact")
    evidence = {
        "classification": "PINNED_IMMUTABLE_RUNTIME_BASE_REQUIRES_STORAGE_INDEPENDENCE",
        "entry_count": len(rows),
        "content_bytes": content_bytes,
        "tree_sha256": _canonical_json_sha256(rows),
    }
    if evidence != KAYA_COMMON_BASE_RUNTIME_CLOSURE:
        raise ExternalLeagueError("Kaya interpreter base closure drifted")
    return evidence


def _verify_kaya_distribution_closure(
    name: str,
    evidence: Mapping[str, object],
    *,
    environment_root: Path,
) -> set[Path]:
    fields = {
        "version", "record", "owned_roots", "file_count", "files_sha256",
        "portable_file_count", "portable_files_sha256", "normalized_record_sha256",
        "record_hashes_verified", "unrecorded_files_absent",
    }
    if (
        set(evidence) != fields
        or evidence.get("version") != KAYA_REQUIRED_DISTRIBUTIONS[name]
        or evidence.get("record_hashes_verified") is not True
        or evidence.get("unrecorded_files_absent") is not True
    ):
        raise ExternalLeagueError(f"Kaya dependency closure evidence is invalid: {name}")
    record = evidence.get("record")
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "entry_count"}:
        raise ExternalLeagueError(f"Kaya dependency RECORD evidence is invalid: {name}")
    record_path = (environment_root / str(record.get("path", ""))).resolve()
    try:
        record_path.relative_to(environment_root)
    except ValueError as exc:
        raise ExternalLeagueError(f"Kaya dependency RECORD escapes isolated environment: {name}") from exc
    if not record_path.is_file() or record.get("sha256") != _sha256_file(record_path):
        raise ExternalLeagueError(f"Kaya dependency RECORD hash mismatches: {name}")
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            record_rows = list(csv.reader(handle))
    except (OSError, csv.Error) as exc:
        raise ExternalLeagueError(f"Kaya dependency RECORD is unreadable: {name}") from exc
    if (
        not record_rows
        or any(len(row) != 3 for row in record_rows)
        or len({row[0] for row in record_rows}) != len(record_rows)
        or record.get("entry_count") != len(record_rows)
    ):
        raise ExternalLeagueError(f"Kaya dependency RECORD row set is invalid: {name}")
    site_packages = record_path.parent.parent.resolve()
    installed_rows: list[dict[str, object]] = []
    portable_rows: list[dict[str, object]] = []
    normalized_record_rows: list[list[str]] = []
    installed_paths: set[Path] = set()
    owned_roots: set[Path] = set()
    for relative, declared_hash, declared_size in record_rows:
        installed = (site_packages / relative).resolve()
        try:
            installed.relative_to(environment_root)
        except ValueError as exc:
            raise ExternalLeagueError(f"Kaya dependency file escapes isolated environment: {name}") from exc
        if not installed.is_file():
            raise ExternalLeagueError(f"Kaya dependency RECORD file is missing: {name}:{relative}")
        data = installed.read_bytes()
        if declared_size and (not declared_size.isdigit() or int(declared_size) != len(data)):
            raise ExternalLeagueError(f"Kaya dependency RECORD size mismatches: {name}:{relative}")
        if declared_hash:
            try:
                algorithm, encoded = declared_hash.split("=", 1)
            except ValueError as exc:
                raise ExternalLeagueError(f"Kaya dependency RECORD digest is invalid: {name}") from exc
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
            if algorithm != "sha256" or encoded != expected:
                raise ExternalLeagueError(f"Kaya dependency RECORD digest mismatches: {name}:{relative}")
        installed_paths.add(installed)
        installed_rows.append({
            "path": str(installed.relative_to(environment_root)),
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        })
        if site_packages in installed.parents and installed != record_path:
            portable_rows.append({
                "path": str(installed.relative_to(site_packages)),
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            })
        if site_packages not in installed.parents:
            normalized_record_rows.append(
                [relative, "<relocatable-generated>", "<relocatable-generated>"]
            )
        elif installed == record_path:
            normalized_record_rows.append([relative, "", ""])
        else:
            normalized_record_rows.append([relative, declared_hash, declared_size])
        if installed == record_path or site_packages in installed.parents:
            relative_to_site = installed.relative_to(site_packages)
            if relative_to_site.parts:
                owned_roots.add((site_packages / relative_to_site.parts[0]).resolve())
    for owned_root in owned_roots:
        candidates = [owned_root] if owned_root.is_file() else list(owned_root.rglob("*"))
        for candidate in candidates:
            if (
                candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix != ".pyc"
                and candidate.resolve() not in installed_paths
            ):
                raise ExternalLeagueError(f"Kaya dependency installed tree has an unrecorded extra file: {name}")
    installed_rows.sort(key=lambda row: str(row["path"]))
    portable_rows.sort(key=lambda row: str(row["path"]))
    normalized_record_rows.sort()
    expected_owned_roots = sorted(str(value.relative_to(environment_root)) for value in owned_roots)
    if (
        evidence.get("owned_roots") != expected_owned_roots
        or evidence.get("file_count") != len(installed_rows)
        or evidence.get("files_sha256") != _canonical_json_sha256(installed_rows)
        or evidence.get("portable_file_count") != len(portable_rows)
        or evidence.get("portable_files_sha256") != _canonical_json_sha256(portable_rows)
        or evidence.get("normalized_record_sha256") != _canonical_json_sha256(normalized_record_rows)
    ):
        raise ExternalLeagueError(f"Kaya dependency full installed-file closure mismatches: {name}")
    immutable = KAYA_DISTRIBUTION_CLOSURES[name]
    if (
        evidence.get("portable_file_count") != immutable["portable_file_count"]
        or evidence.get("portable_files_sha256") != immutable["portable_files_sha256"]
        or evidence.get("normalized_record_sha256") != immutable["normalized_record_sha256"]
    ):
        raise ExternalLeagueError(f"Kaya dependency differs from the frozen wheel closure: {name}")
    return installed_paths


def _execute_kaya_child(
    *,
    mode: str,
    checkout: Path,
    python: Path,
    video: Path,
    direct_manifest: Path | None,
    output_root: Path,
    reuse_count: int,
    timeout_seconds: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True)
    home = output_root / "home"
    cache = output_root / "pycache"
    home.mkdir()
    environment_root = python.parent.parent.resolve()
    environment = _kaya_child_environment(environment_root, home)
    output = output_root / "child-output.json"
    adapter_hash = _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
    command = [
        str(python), "-S", "-X", f"pycache_prefix={cache}", "-c",
        _KAYA_CONFORMANCE_CHILD_SCRIPT, mode, str(checkout), str(video),
        str(direct_manifest) if direct_manifest is not None else "-",
        str(output), adapter_hash, str(reuse_count),
    ]
    started = time.monotonic_ns()
    timed_out = False
    try:
        completed = subprocess.run(
            command, cwd=output_root, capture_output=True, check=False,
            timeout=timeout_seconds, env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        completed = subprocess.CompletedProcess(
            command, 124, exc.stdout or b"",
            (exc.stderr or b"") + b"\nflashpatch: Kaya conformance timeout",
        )
    finished = time.monotonic_ns()
    stdout = output_root / "stdout.bin"
    stderr = output_root / "stderr.bin"
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    process = {
        "command": command,
        "command_sha256": _canonical_json_sha256(command),
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
        "working_directory": str(output_root),
        "started_monotonic_ns": started,
        "finished_monotonic_ns": finished,
        "wall_time_ns": finished - started,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": completed.returncode,
        "stdout": {"path": str(stdout), "sha256": _sha256_file(stdout)},
        "stderr": {"path": str(stderr), "sha256": _sha256_file(stderr)},
        "output": {
            "path": str(output), "exists": output.is_file(),
            "sha256": _sha256_file(output) if output.is_file() else None,
        },
    }
    if timed_out or completed.returncode != 0 or not output.is_file():
        raise ExternalLeagueError(
            f"Kaya {mode} child failed: exit={completed.returncode}, "
            f"stderr_sha256={process['stderr']['sha256']}"
        )
    return process


def _load_kaya_child_output(process: Mapping[str, object]) -> dict[str, object]:
    output_ref = process.get("output")
    if not isinstance(output_ref, Mapping) or set(output_ref) != {"path", "exists", "sha256"}:
        raise ExternalLeagueError("Kaya child output reference is invalid")
    output = Path(str(output_ref.get("path", ""))).resolve()
    if output_ref.get("exists") is not True or not output.is_file() or output_ref.get("sha256") != _sha256_file(output):
        raise ExternalLeagueError("Kaya child output hash mismatches")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("Kaya child output is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ExternalLeagueError("Kaya child output root is invalid")
    return dict(payload)


def _validate_kaya_child_output(
    payload: Mapping[str, object],
    *,
    mode: str,
    reuse_count: int,
    checkout: Path,
    environment_root: Path,
    video: Path,
    conversion: Path,
    direct_manifest: Path | None,
) -> dict[str, object]:
    fields = {
        "schema", "identity", "classification", "adapter_source_sha256", "mode",
        "reuse_count", "upstream", "runtime", "api", "input", "capture_runs",
        "results", "direct_conversion", "claim_boundary",
    }
    if (
        set(payload) != fields
        or payload.get("schema") != KAYA_CONFORMANCE_CHILD_SCHEMA
        or payload.get("identity") != KAYA_PROTOTYPE_ID
        or payload.get("classification") != "UNSCORED_CONFORMANCE_ONLY"
        or payload.get("adapter_source_sha256") != _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
        or payload.get("mode") != mode
        or payload.get("reuse_count") != reuse_count
        or payload.get("claim_boundary") != {
            "scoreable": False,
            "population_authorized": False,
            "participant_status": "UNSCORED_PROTOTYPE",
        }
    ):
        raise ExternalLeagueError("Kaya child identity or claim boundary is invalid")
    expected_upstream = {
        "revision": KAYA_SOURCE_REVISION,
        "tree": KAYA_SOURCE_TREE,
        "source_hashes": dict(KAYA_REQUIRED_SOURCE_HASHES),
        "license": "BSD-3-Clause",
    }
    if payload.get("upstream") != expected_upstream:
        raise ExternalLeagueError("Kaya child upstream source identity drifted")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python_executable", "python_executable_sha256", "python_version",
        "python_version_info", "sys_prefix", "base_prefix", "environment_root",
        "site_packages", "no_site", "sys_path", "dependencies", "imported_modules",
        "import_census", "runtime_base", "loaded_modules",
    }:
        raise ExternalLeagueError("Kaya child runtime evidence fields are invalid")
    python_executable = Path(str(runtime.get("python_executable", ""))).resolve()
    interpreter_root = python_executable.parent.parent.resolve()
    version_info = runtime.get("python_version_info")
    if (
        not isinstance(version_info, list) or len(version_info) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in version_info)
        or version_info[0] != 3 or not 8 <= version_info[1] <= 10
    ):
        raise ExternalLeagueError("Kaya child Python version or isolated prefix is invalid")
    expected_site_packages = (
        environment_root / "lib" / f"python{version_info[0]}.{version_info[1]}" / "site-packages"
    )
    if (
        runtime.get("sys_prefix") != str(interpreter_root)
        or runtime.get("base_prefix") != str(interpreter_root)
        or runtime.get("environment_root") != str(environment_root)
        or runtime.get("site_packages") != str(expected_site_packages)
        or runtime.get("no_site") is not True
    ):
        raise ExternalLeagueError("Kaya child Python version or isolated prefix is invalid")
    runtime_base = runtime.get("runtime_base")
    if (
        not isinstance(runtime_base, Mapping)
        or dict(runtime_base) != _kaya_runtime_base_evidence(interpreter_root)
    ):
        raise ExternalLeagueError("Kaya child interpreter base closure is invalid")
    if (
        not python_executable.is_file()
        or runtime.get("python_executable_sha256") != _sha256_file(python_executable)
        or runtime.get("python_executable_sha256") != KAYA_PYTHON_SHA256
    ):
        raise ExternalLeagueError("Kaya child Python executable hash mismatches")
    sys_path = runtime.get("sys_path")
    if not isinstance(sys_path, list) or not all(isinstance(item, str) for item in sys_path):
        raise ExternalLeagueError("Kaya child sys.path evidence is invalid")
    allowed_import_roots = (checkout.resolve(), environment_root.resolve(), interpreter_root)
    if not sys_path or Path(sys_path[0]).resolve() != checkout.resolve():
        raise ExternalLeagueError("Kaya child checkout is not first on sys.path")
    for item in sys_path:
        if item == "":
            raise ExternalLeagueError("Kaya child sys.path retains the current working directory")
        imported_root = Path(item).resolve()
        if not any(imported_root == allowed or allowed in imported_root.parents for allowed in allowed_import_roots):
            raise ExternalLeagueError("Kaya child sys.path escapes the checkout or isolated runtime")
    dependencies = runtime.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(KAYA_REQUIRED_DISTRIBUTIONS):
        raise ExternalLeagueError("Kaya child dependency population is invalid")
    recorded_distribution_paths: dict[str, set[Path]] = {}
    for name in KAYA_REQUIRED_DISTRIBUTIONS:
        evidence = dependencies.get(name)
        if not isinstance(evidence, Mapping):
            raise ExternalLeagueError(f"Kaya child dependency evidence is invalid: {name}")
        recorded_distribution_paths[name] = _verify_kaya_distribution_closure(
            name, evidence, environment_root=environment_root
        )
    imported_modules = runtime.get("imported_modules")
    if (
        not isinstance(imported_modules, Mapping)
        or set(imported_modules) != set(KAYA_REQUIRED_DISTRIBUTIONS)
    ):
        raise ExternalLeagueError("Kaya child imported module evidence is invalid")
    site_packages: Path | None = None
    for distribution_name, reference in imported_modules.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ExternalLeagueError(
                f"Kaya child imported module reference is invalid: {distribution_name}"
            )
        module_path = Path(str(reference.get("path", ""))).resolve()
        try:
            module_path.relative_to(environment_root)
        except ValueError as exc:
            raise ExternalLeagueError(
                f"Kaya child imported module escapes isolated environment: {distribution_name}"
            ) from exc
        if not module_path.is_file() or reference.get("sha256") != _sha256_file(module_path):
            raise ExternalLeagueError(
                f"Kaya child imported module hash mismatches: {distribution_name}"
            )
        immutable_module = KAYA_DISTRIBUTION_CLOSURES[distribution_name]
        if (
            module_path not in recorded_distribution_paths[distribution_name]
            or str(module_path.relative_to(environment_root)) != immutable_module["module_path"]
            or reference.get("sha256") != immutable_module["module_sha256"]
        ):
            raise ExternalLeagueError(
                f"Kaya child imported module is not owned by its frozen RECORD: {distribution_name}"
            )
        module_site_packages = next(
            (parent for parent in module_path.parents if parent.name == "site-packages"),
            None,
        )
        if module_site_packages is None:
            raise ExternalLeagueError(
                f"Kaya child imported module lacks an isolated site-packages root: {distribution_name}"
            )
        if site_packages is None:
            site_packages = module_site_packages
        elif module_site_packages != site_packages:
            raise ExternalLeagueError("Kaya child dependencies came from multiple site-packages roots")
    if site_packages is None or site_packages != expected_site_packages:
        raise ExternalLeagueError("Kaya child isolated site-packages root is missing")
    census = runtime.get("import_census")
    if not isinstance(census, Mapping) or set(census) != {
        "rows", "rows_sha256", "distributions", "module_count",
        "all_site_modules_record_owned",
    }:
        raise ExternalLeagueError("Kaya child imported distribution census fields are invalid")
    census_rows = census.get("rows")
    if not isinstance(census_rows, list):
        raise ExternalLeagueError("Kaya child imported distribution census rows are invalid")
    observed_modules: list[str] = []
    observed_distributions: set[str] = set()
    for row in census_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "module", "path", "distribution", "sha256",
        }:
            raise ExternalLeagueError("Kaya child imported distribution census row is invalid")
        module_name = row.get("module")
        distribution_name = row.get("distribution")
        if (
            not isinstance(module_name, str)
            or not module_name
            or distribution_name not in KAYA_REQUIRED_DISTRIBUTIONS
        ):
            raise ExternalLeagueError("Kaya child imported distribution census identity is invalid")
        module_path = (environment_root / str(row.get("path", ""))).resolve()
        try:
            module_path.relative_to(environment_root)
        except ValueError as exc:
            raise ExternalLeagueError(
                "Kaya child imported distribution census path escapes the environment"
            ) from exc
        if (
            not module_path.is_file()
            or row.get("sha256") != _sha256_file(module_path)
            or module_path not in recorded_distribution_paths[str(distribution_name)]
        ):
            raise ExternalLeagueError(
                f"Kaya child imported module is not RECORD-owned: {module_name}"
            )
        observed_modules.append(module_name)
        observed_distributions.add(str(distribution_name))
    if (
        observed_modules != sorted(observed_modules)
        or len(set(observed_modules)) != len(observed_modules)
        or census.get("rows_sha256") != _canonical_json_sha256(census_rows)
        or census.get("module_count") != len(census_rows)
        or census.get("distributions") != sorted(KAYA_REQUIRED_DISTRIBUTIONS)
        or observed_distributions != set(KAYA_REQUIRED_DISTRIBUTIONS)
        or census.get("all_site_modules_record_owned") is not True
    ):
        raise ExternalLeagueError("Kaya actual imported distribution census differs from the frozen closure")
    loaded_modules = runtime.get("loaded_modules")
    if not isinstance(loaded_modules, Mapping) or set(loaded_modules) != {
        "rows", "rows_sha256", "module_count", "shared_object_count", "import_hooks",
    }:
        raise ExternalLeagueError("Kaya loaded-module census fields are invalid")
    loaded_rows = loaded_modules.get("rows")
    if not isinstance(loaded_rows, list):
        raise ExternalLeagueError("Kaya loaded-module census rows are invalid")
    loaded_names: list[str] = []
    shared_object_count = 0
    roots = {
        "upstream_source": checkout.resolve(),
        "site_packages": expected_site_packages,
        "interpreter_base": interpreter_root,
    }
    for row in loaded_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "module", "classification", "path", "sha256", "shared_object",
        }:
            raise ExternalLeagueError("Kaya loaded-module census row is invalid")
        module_name = row.get("module")
        classification = row.get("classification")
        if (
            not isinstance(module_name, str)
            or not module_name
            or classification not in roots
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("shared_object"), bool)
        ):
            raise ExternalLeagueError("Kaya loaded-module census identity is invalid")
        root = roots[str(classification)]
        module_path = (root / str(row["path"])).resolve()
        try:
            module_path.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError(
                f"Kaya loaded module escapes its frozen root: {module_name}"
            ) from exc
        if (
            not module_path.is_file()
            or row.get("sha256") != _sha256_file(module_path)
            or row.get("shared_object") is not (module_path.suffix == ".so")
        ):
            raise ExternalLeagueError(f"Kaya loaded module artifact drifted: {module_name}")
        loaded_names.append(module_name)
        shared_object_count += int(bool(row["shared_object"]))
    if (
        loaded_names != sorted(loaded_names)
        or len(set(loaded_names)) != len(loaded_names)
        or loaded_modules.get("rows_sha256") != _canonical_json_sha256(loaded_rows)
        or loaded_modules.get("module_count") != len(loaded_rows)
        or loaded_modules.get("shared_object_count") != shared_object_count
        or loaded_modules.get("import_hooks") != KAYA_IMPORT_HOOKS
    ):
        raise ExternalLeagueError("Kaya loaded modules or import hooks differ from the frozen runtime")
    api = payload.get("api")
    if not isinstance(api, Mapping) or set(api) != {
        "GuidelineProcess.analyse_file", "Display", "CustomVideo.frame_intervals", "function_objects",
        "guideline_object_module", "guideline_object_qualname",
        "function_objects_module", "pipeline", "pipeline_sha256", "display_defaults",
    }:
        raise ExternalLeagueError("Kaya child upstream API evidence fields are invalid")
    expected_api = {
        "GuidelineProcess.analyse_file": (
            "PhotosensitivitySafetyEngine.engine.analysis", "GuidelineProcess.analyse_file",
            KAYA_REQUIRED_SOURCE_HASHES["PhotosensitivitySafetyEngine/engine/analysis.py"],
        ),
        "Display": (
            "PhotosensitivitySafetyEngine.engine.analysis", "Display",
            KAYA_REQUIRED_SOURCE_HASHES["PhotosensitivitySafetyEngine/engine/analysis.py"],
        ),
        "CustomVideo.frame_intervals": (
            "custom_video", "CustomVideo.frame_intervals", KAYA_REQUIRED_SOURCE_HASHES["custom_video.py"],
        ),
        "function_objects": (
            "PhotosensitivitySafetyEngine.guidelines.w3c", "<lambda>",
            KAYA_REQUIRED_SOURCE_HASHES["PhotosensitivitySafetyEngine/guidelines/w3c.py"],
        ),
    }
    for label, (module, qualname, module_hash) in expected_api.items():
        evidence = api.get(label)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("module") != module or evidence.get("qualname") != qualname
            or evidence.get("module_sha256") != module_hash
            or evidence.get("callable_source_sha256") != KAYA_CALLABLE_SOURCE_HASHES[label]
        ):
            raise ExternalLeagueError(f"Kaya child did not bind the unmodified upstream API: {label}")
        module_path = Path(str(evidence.get("module_path", ""))).resolve()
        try:
            module_path.relative_to(checkout)
        except ValueError as exc:
            raise ExternalLeagueError(f"Kaya child upstream API path escapes checkout: {label}") from exc
    if (
        api.get("guideline_object_module") != "PhotosensitivitySafetyEngine.engine.analysis"
        or api.get("guideline_object_qualname") != "GuidelineProcess"
        or api.get("function_objects_module") != "PhotosensitivitySafetyEngine.guidelines.w3c"
        or api.get("pipeline_sha256") != KAYA_PIPELINE_SHA256
        or api.get("pipeline_sha256") != _canonical_json_sha256(api.get("pipeline"))
        or api.get("display_defaults") != {
            "display_resolution": [1024, 768], "display_diameter": 16,
            "display_distance": 24, "frame_rate_before_input": 30,
            "candelas": 200, "speedup": 10,
            "expected_analysis_resolution": [102, 76],
        }
    ):
        raise ExternalLeagueError("Kaya child Display, object, or pipeline identity drifted")
    input_ref = payload.get("input")
    if not isinstance(input_ref, Mapping) or set(input_ref) != {
        "video", "video_sha256", "direct_manifest", "direct_manifest_sha256",
    }:
        raise ExternalLeagueError("Kaya child input binding is invalid")
    if input_ref.get("video") != str(video) or input_ref.get("video_sha256") != _sha256_file(video):
        raise ExternalLeagueError("Kaya child canonical video binding mismatches")
    if mode == "native":
        if input_ref.get("direct_manifest") is not None or input_ref.get("direct_manifest_sha256") is not None:
            raise ExternalLeagueError("Kaya native child falsely declares a direct input")
        direct_payload = None
    else:
        if direct_manifest is None or input_ref.get("direct_manifest") != str(direct_manifest) or input_ref.get("direct_manifest_sha256") != _sha256_file(direct_manifest):
            raise ExternalLeagueError("Kaya direct child manifest binding mismatches")
        direct_payload = _load_kaya_direct_input_manifest(
            direct_manifest, expected_video=video, expected_conversion=conversion,
        )
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    source = Path(str(contract["renderer_source"]["path"])).resolve()
    with np.load(source) as archive:
        rgb_frames = np.asarray(archive["frames"])
    expected_bgr_ledger = [
        {
            "index": index, "shape": list(rgb_frames[index].shape),
            "pixel_format": "bgr24",
            "bgr_sha256": _sha256_bytes(np.ascontiguousarray(rgb_frames[index][..., ::-1]).tobytes()),
        }
        for index in range(len(rgb_frames))
    ]
    capture_runs = payload.get("capture_runs")
    results = payload.get("results")
    if not isinstance(capture_runs, list) or not isinstance(results, list) or len(capture_runs) != reuse_count or len(results) != reuse_count:
        raise ExternalLeagueError("Kaya child capture or result invocation count mismatches")
    for capture in capture_runs:
        if (
            not isinstance(capture, Mapping)
            or set(capture) != {"ledger", "observed_fps", "observed_frame_count", "ledger_sha256"}
            or capture.get("observed_fps") != 60.0
            or capture.get("observed_frame_count") != len(rgb_frames)
            or capture.get("ledger") != expected_bgr_ledger
            or capture.get("ledger_sha256") != _canonical_json_sha256(expected_bgr_ledger)
        ):
            raise ExternalLeagueError("Kaya child pre-consumption BGR ledger differs from canonical RGB")
    for result in results:
        if not isinstance(result, Mapping) or set(result) != {"raw", "interval_tuples", "interval_semantics"}:
            raise ExternalLeagueError("Kaya child raw result fields are invalid")
        raw = result.get("raw")
        if (
            not isinstance(raw, Mapping) or set(raw) != {"General Flashes", "Red Flashes"}
            or any(
                not isinstance(values, list) or len(values) != len(rgb_frames)
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in values)
                for values in raw.values()
            )
            or result.get("interval_semantics") != "unmodified_CustomVideo.frame_intervals_including_eof_open_both_bug"
            or not isinstance(result.get("interval_tuples"), list)
        ):
            raise ExternalLeagueError("Kaya child raw arrays or interval semantics are invalid")
        for interval in result["interval_tuples"]:
            if (
                not isinstance(interval, list) or len(interval) != 3
                or interval[0] not in {"general", "red", "both"}
                or any(isinstance(value, bool) or not isinstance(value, int) for value in interval[1:])
                or not 0 <= interval[1] < interval[2] <= len(rgb_frames)
            ):
                raise ExternalLeagueError("Kaya child upstream interval tuple is invalid")
    direct_conversion = payload.get("direct_conversion")
    if mode == "native":
        if direct_conversion is not None:
            raise ExternalLeagueError("Kaya native child declares a direct color conversion")
    else:
        if not isinstance(direct_payload, Mapping) or not isinstance(direct_conversion, Mapping) or set(direct_conversion) != {"operation", "pairs", "pairs_sha256"}:
            raise ExternalLeagueError("Kaya direct child color conversion evidence is invalid")
        pairs = [
            {
                "index": index,
                "rgb_sha256": direct_payload["ledger"][index]["rgb_sha256"],
                "bgr_sha256": expected_bgr_ledger[index]["bgr_sha256"],
            }
            for index in range(len(rgb_frames))
        ]
        if (
            direct_conversion.get("operation") != "rgb_to_bgr_channel_reverse_once"
            or direct_conversion.get("pairs") != pairs
            or direct_conversion.get("pairs_sha256") != _canonical_json_sha256(pairs)
        ):
            raise ExternalLeagueError("Kaya direct RGB to BGR conversion is not exactly one channel reversal")
    return dict(payload)


def _kaya_cross_environment_projection(
    payload: Mapping[str, object],
    *,
    environment_root: Path,
) -> dict[str, object]:
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExternalLeagueError("Kaya child runtime evidence is missing")
    dependencies = runtime.get("dependencies")
    imported_modules = runtime.get("imported_modules")
    if not isinstance(dependencies, Mapping) or not isinstance(imported_modules, Mapping):
        raise ExternalLeagueError("Kaya child portable runtime evidence is missing")
    portable_dependencies: dict[str, dict[str, object]] = {}
    for name in sorted(KAYA_REQUIRED_DISTRIBUTIONS):
        evidence = dependencies.get(name)
        if not isinstance(evidence, Mapping):
            raise ExternalLeagueError(f"Kaya portable dependency evidence is invalid: {name}")
        portable_dependencies[name] = {
            "version": evidence.get("version"),
            "portable_file_count": evidence.get("portable_file_count"),
            "portable_files_sha256": evidence.get("portable_files_sha256"),
            "normalized_record_sha256": evidence.get("normalized_record_sha256"),
            "record_hashes_verified": evidence.get("record_hashes_verified"),
            "unrecorded_files_absent": evidence.get("unrecorded_files_absent"),
        }
    portable_modules: dict[str, dict[str, object]] = {}
    for name in sorted(KAYA_REQUIRED_DISTRIBUTIONS):
        reference = imported_modules.get(name)
        if not isinstance(reference, Mapping):
            raise ExternalLeagueError(f"Kaya portable imported module evidence is invalid: {name}")
        module_path = Path(str(reference.get("path", ""))).resolve()
        try:
            relative = module_path.relative_to(environment_root)
        except ValueError as exc:
            raise ExternalLeagueError(
                f"Kaya portable imported module escapes its environment: {name}"
            ) from exc
        portable_modules[name] = {
            "path": str(relative),
            "sha256": reference.get("sha256"),
        }
    return {
        **{key: payload[key] for key in payload if key != "runtime"},
        "runtime": {
            "python_executable_sha256": runtime.get("python_executable_sha256"),
            "python_version": runtime.get("python_version"),
            "python_version_info": runtime.get("python_version_info"),
            "no_site": runtime.get("no_site"),
            "dependencies": portable_dependencies,
            "imported_modules": portable_modules,
            "import_census": runtime.get("import_census"),
            "runtime_base": runtime.get("runtime_base"),
            "loaded_modules": runtime.get("loaded_modules"),
        },
    }


def _kaya_dependency_file_inodes(
    payload: Mapping[str, object],
    *,
    environment_root: Path,
) -> dict[tuple[int, int], list[str]]:
    runtime = payload.get("runtime")
    dependencies = runtime.get("dependencies") if isinstance(runtime, Mapping) else None
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(KAYA_REQUIRED_DISTRIBUTIONS):
        raise ExternalLeagueError("Kaya dependency storage evidence is invalid")
    identities: dict[tuple[int, int], list[str]] = {}
    for name in sorted(KAYA_REQUIRED_DISTRIBUTIONS):
        evidence = dependencies.get(name)
        record = evidence.get("record") if isinstance(evidence, Mapping) else None
        if not isinstance(record, Mapping):
            raise ExternalLeagueError(f"Kaya dependency storage RECORD is invalid: {name}")
        record_path = (environment_root / str(record.get("path", ""))).resolve()
        try:
            record_path.relative_to(environment_root)
            with record_path.open("r", encoding="utf-8", newline="") as handle:
                record_rows = list(csv.reader(handle))
        except (ValueError, OSError, csv.Error) as exc:
            raise ExternalLeagueError(
                f"Kaya dependency storage RECORD cannot be audited: {name}"
            ) from exc
        site_packages = record_path.parent.parent.resolve()
        for row in record_rows:
            if len(row) != 3:
                raise ExternalLeagueError(f"Kaya dependency storage RECORD row is invalid: {name}")
            installed = (site_packages / row[0]).resolve()
            try:
                installed.relative_to(environment_root)
            except ValueError as exc:
                raise ExternalLeagueError(
                    f"Kaya dependency storage file escapes its environment: {name}"
                ) from exc
            if not installed.is_file():
                raise ExternalLeagueError(f"Kaya dependency storage file is missing: {name}")
            observed = installed.stat()
            identities.setdefault((observed.st_dev, observed.st_ino), []).append(
                f"{name}:{installed.relative_to(environment_root)}"
            )
    return identities


def _kaya_runtime_storage_inodes(
    payload: Mapping[str, object],
    *,
    environment_root: Path,
) -> tuple[dict[tuple[int, int], list[str]], dict[str, int]]:
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExternalLeagueError("Kaya runtime storage evidence is invalid")
    interpreter_root = Path(str(runtime.get("base_prefix", ""))).resolve()
    site_packages = Path(str(runtime.get("site_packages", ""))).resolve()
    python_executable = Path(str(runtime.get("python_executable", ""))).resolve()
    if (
        not interpreter_root.is_dir()
        or not site_packages.is_dir()
        or not python_executable.is_file()
        or runtime.get("runtime_base") != _kaya_runtime_base_evidence(interpreter_root)
    ):
        raise ExternalLeagueError("Kaya interpreter/runtime storage closure is invalid")
    try:
        site_packages.relative_to(environment_root)
        python_executable.relative_to(interpreter_root)
    except ValueError as exc:
        raise ExternalLeagueError("Kaya runtime storage escapes its frozen roots") from exc

    identities = _kaya_dependency_file_inodes(payload, environment_root=environment_root)
    counts = {
        "dependency_recorded_files": sum(len(rows) for rows in identities.values()),
        "runtime_base_entries": 0,
        "loaded_runtime_modules": 0,
        "loaded_shared_objects": 0,
        "storage_roots": 0,
    }

    def bind(path: Path, label: str) -> None:
        observed = path.stat()
        identities.setdefault((observed.st_dev, observed.st_ino), []).append(label)

    for root, label in (
        (environment_root, "environment_root"),
        (site_packages, "site_packages_root"),
        (interpreter_root, "interpreter_base_root"),
    ):
        if not root.is_dir():
            raise ExternalLeagueError(f"Kaya runtime storage root is unavailable: {label}")
        bind(root, label)
        counts["storage_roots"] += 1
    bind(python_executable, "python_executable")

    for path in sorted(interpreter_root.rglob("*")):
        if path.is_symlink() or path.is_file():
            resolved = path.resolve()
            try:
                resolved.relative_to(interpreter_root)
            except ValueError as exc:
                raise ExternalLeagueError("Kaya interpreter storage link escapes its root") from exc
            if not resolved.is_file():
                raise ExternalLeagueError("Kaya interpreter storage entry is unavailable")
            bind(resolved, f"interpreter_base:{path.relative_to(interpreter_root)}")
            counts["runtime_base_entries"] += 1

    loaded_modules = runtime.get("loaded_modules")
    loaded_rows = loaded_modules.get("rows") if isinstance(loaded_modules, Mapping) else None
    if not isinstance(loaded_rows, list):
        raise ExternalLeagueError("Kaya loaded-module storage evidence is invalid")
    roots = {
        "site_packages": site_packages,
        "interpreter_base": interpreter_root,
    }
    for row in loaded_rows:
        if not isinstance(row, Mapping):
            raise ExternalLeagueError("Kaya loaded-module storage row is invalid")
        classification = row.get("classification")
        if classification == "upstream_source":
            continue
        root = roots.get(str(classification))
        if root is None:
            raise ExternalLeagueError("Kaya loaded-module storage class is invalid")
        path = (root / str(row.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ExternalLeagueError("Kaya loaded-module storage escapes its root") from exc
        if not path.is_file() or row.get("sha256") != _sha256_file(path):
            raise ExternalLeagueError("Kaya loaded-module storage hash mismatches")
        bind(path, f"loaded_module:{row.get('module')}")
        counts["loaded_runtime_modules"] += 1
        counts["loaded_shared_objects"] += int(bool(row.get("shared_object")))
    return identities, counts


def _verify_kaya_replay_storage_independence(
    primary_payload: Mapping[str, object],
    replay_payload: Mapping[str, object],
    *,
    primary_environment_root: Path,
    replay_environment_root: Path,
) -> dict[str, object]:
    primary, primary_counts = _kaya_runtime_storage_inodes(
        primary_payload, environment_root=primary_environment_root
    )
    replay, replay_counts = _kaya_runtime_storage_inodes(
        replay_payload, environment_root=replay_environment_root
    )
    shared = sorted(set(primary) & set(replay))
    if shared:
        first = shared[0]
        raise ExternalLeagueError(
            "Kaya fresh replay runtime storage is aliased across environments: "
            f"primary={primary[first][0]}, replay={replay[first][0]}"
        )
    return {
        "primary_identity_count": len(primary),
        "replay_identity_count": len(replay),
        "primary_counts": primary_counts,
        "replay_counts": replay_counts,
        "shared_inode_count": 0,
    }


def _kaya_semantic_conformance_failures(fixtures: Mapping[str, Mapping[str, object]]) -> list[str]:
    failures: list[str] = []
    if set(fixtures) != set(KAYA_REQUIRED_FIXTURE_IDS):
        return ["required_fixture_population_missing"]
    for fixture_id in KAYA_REQUIRED_FIXTURE_IDS:
        row = fixtures[fixture_id]
        native = row["native_output"]
        direct = row["direct_output"]
        native_result = native["results"][0]
        if any(result != native_result for result in direct["results"]):
            failures.append(f"native_direct_exact_result_mismatch:{fixture_id}")
        if direct["capture_runs"][0]["ledger"] != native["capture_runs"][0]["ledger"]:
            failures.append(f"native_direct_preconsumption_bgr_mismatch:{fixture_id}")
    safe = fixtures["safe"]["native_output"]["results"][0]
    if safe["interval_tuples"] or max(safe["raw"]["General Flashes"] + safe["raw"]["Red Flashes"]) > 3:
        failures.append("safe_fixture_not_safe_under_upstream")
    channel = fixtures["rgb-channel-trap"]["native_output"]["results"][0]["raw"]
    if channel["General Flashes"] == channel["Red Flashes"] or max(channel["Red Flashes"]) <= 0:
        failures.append("rgb_channel_trap_not_discriminating")
    threshold = fixtures["flash-threshold"]["native_output"]["results"][0]
    threshold_general = threshold["raw"]["General Flashes"]
    first_over_threshold = next(
        (index for index, value in enumerate(threshold_general) if value > 3),
        None,
    )
    if (
        max(threshold_general) <= 3
        or 3.0 not in threshold_general
        or 3.5 not in threshold_general
        or threshold["interval_tuples"] != [["general", first_over_threshold, len(threshold_general)]]
    ):
        failures.append("flash_threshold_fixture_did_not_cross_upstream_threshold")
    history = {
        count: fixtures[f"history-{count}"]["native_output"]["results"][0]["raw"]
        for count in (59, 60, 61)
    }
    for key in ("General Flashes", "Red Flashes"):
        if len(history[59][key]) != 59 or len(history[60][key]) != 60 or len(history[61][key]) != 61:
            failures.append(f"history_boundary_length_invalid:{key}")
        elif history[59][key] != history[60][key][:59] or history[60][key] != history[61][key][:60]:
            failures.append(f"history_boundary_prefix_drift:{key}")
    shape = fixtures["letterbox"]["input"]["shape"]
    if not isinstance(shape, list) or len(shape) != 4 or shape[2] * 3 == shape[1] * 4:
        failures.append("letterbox_fixture_does_not_exercise_aspect_fit")
    state = fixtures["state-reuse"]["direct_output"]["results"]
    if len(state) != 2 or state[0] != state[1]:
        failures.append("state_reuse_changed_upstream_result")
    elif (
        state[0]["raw"]["General Flashes"][-1] <= 3
        or state[0]["raw"]["Red Flashes"][-1] <= 3
        or state[0]["interval_tuples"]
    ):
        failures.append("eof_open_both_upstream_bug_not_exercised_verbatim")
    return failures


def _kaya_pre_replay_status() -> str:
    """The execution receipt is never a fresh-replay verification receipt."""
    return "NOT_VERIFIED"


def execute_kaya_conformance_prototype(
    checkout: Path | str,
    python_executable: Path | str,
    fixtures: Mapping[str, Mapping[str, object]],
    output_root: Path | str,
    *,
    timeout_seconds: int = 180,
) -> dict[str, object]:
    """Execute native/direct Kaya conformance without entering the L7 population."""
    checkout_path = Path(checkout).resolve()
    supplied_python = Path(python_executable).absolute()
    resolved_python = supplied_python.resolve()
    environment_root = supplied_python.parent.parent.resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Kaya conformance output already exists: {root}")
    if set(fixtures) != set(KAYA_REQUIRED_FIXTURE_IDS):
        raise ExternalLeagueError("Kaya conformance requires the complete frozen fixture population")
    if not supplied_python.is_file() or supplied_python.parent.name != "bin" or not resolved_python.is_file():
        raise ExternalLeagueError("Kaya isolated Python executable is unavailable")
    if _sha256_file(resolved_python) != KAYA_PYTHON_SHA256:
        raise ExternalLeagueError("Kaya isolated Python executable differs from the frozen toolchain")
    provenance = _audit_kaya_source_checkout(checkout_path)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ExternalLeagueError("Kaya conformance timeout must be a positive integer")
    root.mkdir(parents=True)
    fixture_rows: dict[str, dict[str, object]] = {}
    for fixture_id in KAYA_REQUIRED_FIXTURE_IDS:
        declared = fixtures[fixture_id]
        if not isinstance(declared, Mapping) or declared.get("fixture_id") != fixture_id:
            raise ExternalLeagueError(f"Kaya fixture identity is invalid: {fixture_id}")
        video = Path(str(declared.get("video", ""))).resolve()
        conversion = Path(str(declared.get("conversion_receipt", ""))).resolve()
        _, contract = _canonical_decoder_timeline_contract(video, conversion)
        if contract.get("fps") != 60 or declared.get("fps") != 60:
            raise ExternalLeagueError(f"Kaya fixture is not exact 60 CFR: {fixture_id}")
        fixture_root = root / fixture_id
        fixture_root.mkdir()
        direct_payload = _materialize_kaya_direct_input(
            fixture_id, video, conversion, fixture_root / "direct-input",
        )
        direct_manifest = Path(str(direct_payload["manifest"])).resolve()
        native_process = _execute_kaya_child(
            mode="native", checkout=checkout_path, python=supplied_python,
            video=video, direct_manifest=None, output_root=fixture_root / "native",
            reuse_count=1, timeout_seconds=timeout_seconds,
        )
        direct_process = _execute_kaya_child(
            mode="direct", checkout=checkout_path, python=supplied_python,
            video=video, direct_manifest=direct_manifest, output_root=fixture_root / "direct",
            reuse_count=2 if fixture_id == "state-reuse" else 1,
            timeout_seconds=timeout_seconds,
        )
        native_output = _validate_kaya_child_output(
            _load_kaya_child_output(native_process), mode="native", reuse_count=1,
            checkout=checkout_path, environment_root=environment_root,
            video=video, conversion=conversion, direct_manifest=None,
        )
        direct_output = _validate_kaya_child_output(
            _load_kaya_child_output(direct_process), mode="direct",
            reuse_count=2 if fixture_id == "state-reuse" else 1,
            checkout=checkout_path, environment_root=environment_root,
            video=video, conversion=conversion, direct_manifest=direct_manifest,
        )
        fixture_rows[fixture_id] = {
            "fixture_id": fixture_id,
            "classification": "CONTROLLED_CONFORMANCE_NOT_NATURAL_NOT_SCORING",
            "input": {
                "video": str(video), "video_sha256": _sha256_file(video),
                "conversion_receipt": str(conversion),
                "conversion_receipt_sha256": _sha256_file(conversion),
                "frame_count": contract["frame_count"], "fps": contract["fps"],
                "shape": contract["shape"], "frame_map_sha256": contract["frame_map_sha256"],
            },
            "direct_input": {"path": str(direct_manifest), "sha256": _sha256_file(direct_manifest)},
            "native_process": native_process,
            "direct_process": direct_process,
            "native_output": native_output,
            "direct_output": direct_output,
        }
    failures = _kaya_semantic_conformance_failures(fixture_rows)
    receipt = {
        "schema": KAYA_CONFORMANCE_PROTOTYPE_SCHEMA,
        "identity": KAYA_PROTOTYPE_ID,
        "classification": "UNSCORED_CONFORMANCE_ONLY",
        "upstream": provenance,
        "adapter_source_sha256": _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8")),
        "environment": {
            "root": str(environment_root), "python": str(supplied_python),
            "python_sha256": _sha256_file(resolved_python),
            "required_distributions": dict(KAYA_REQUIRED_DISTRIBUTIONS),
        },
        "fixture_ids": list(KAYA_REQUIRED_FIXTURE_IDS),
        "fixtures": fixture_rows,
        "conformance": {
            "native_api": "unmodified_GuidelineProcess.analyse_file",
            "direct_api": "same_unmodified_GuidelineProcess.analyse_file_with_receipt_bound_DirectCapture",
            "rgb_to_bgr": "exactly_one_channel_reverse_before_upstream_consumption",
            "result_equality": "exact_raw_arrays_and_exact_upstream_interval_tuples_no_normalization",
            "eof_open_both_bug": "preserved_unmodified",
            "failures": failures,
        },
        # Semantic agreement is pre-replay evidence only.
        "status": _kaya_pre_replay_status(),
        "claim_status": "UNSCORED_CONFORMANCE_ONLY",
        "scoreable": False,
        "population_authorized": False,
        "population_mutated": False,
        "scoreable_blockers": [
            "fixed_L7_population_unchanged", "independent_execution_witness_missing",
            "natural_case_and_independent_gold_missing", "fair_three_repeat_runtime_missing",
        ],
    }
    receipt_path = root / "kaya-conformance-prototype-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def _finalize_kaya_verification(
    stored: Mapping[str, object],
    *,
    expected_status: str,
    failures: Sequence[str],
    fresh_replay: bool,
) -> dict[str, object]:
    result = dict(stored)
    if not fresh_replay:
        result["status"] = "NOT_VERIFIED"
        result["verification_status"] = "NOT_VERIFIED"
        result["verification_blockers"] = ["fresh_replay_disabled"]
    else:
        result["status"] = expected_status
        result["verification_status"] = expected_status
        result["verification_blockers"] = [] if expected_status == "VERIFIED" else list(failures)
    result["fresh_replay_verified"] = fresh_replay and expected_status == "VERIFIED"
    return result


def verify_kaya_conformance_prototype(
    receipt_path: Path | str,
    *,
    fresh_replay: bool = True,
    replay_python_executable: Path | str | None = None,
) -> dict[str, object]:
    """Recheck receipt artifacts and replay with a separately built runtime."""
    path = Path(receipt_path).resolve()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("Kaya conformance receipt is unreadable") from exc
    fields = {
        "schema", "identity", "classification", "upstream", "adapter_source_sha256",
        "environment", "fixture_ids", "fixtures", "conformance", "status",
        "claim_status", "scoreable", "population_authorized", "population_mutated",
        "scoreable_blockers",
    }
    if (
        not isinstance(stored, Mapping) or set(stored) != fields
        or stored.get("schema") != KAYA_CONFORMANCE_PROTOTYPE_SCHEMA
        or stored.get("identity") != KAYA_PROTOTYPE_ID
        or stored.get("classification") != "UNSCORED_CONFORMANCE_ONLY"
        or stored.get("adapter_source_sha256") != _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
        or stored.get("claim_status") != "UNSCORED_CONFORMANCE_ONLY"
        or stored.get("scoreable") is not False
        or stored.get("population_authorized") is not False
        or stored.get("population_mutated") is not False
        or stored.get("fixture_ids") != list(KAYA_REQUIRED_FIXTURE_IDS)
        or stored.get("scoreable_blockers") != [
            "fixed_L7_population_unchanged", "independent_execution_witness_missing",
            "natural_case_and_independent_gold_missing", "fair_three_repeat_runtime_missing",
        ]
    ):
        raise ExternalLeagueError("Kaya conformance receipt identity or claim boundary is invalid")
    fixture_rows = stored.get("fixtures")
    if not isinstance(fixture_rows, Mapping) or set(fixture_rows) != set(KAYA_REQUIRED_FIXTURE_IDS):
        raise ExternalLeagueError("Kaya conformance receipt fixture population is invalid")
    first = fixture_rows[KAYA_REQUIRED_FIXTURE_IDS[0]]
    if not isinstance(first, Mapping):
        raise ExternalLeagueError("Kaya conformance first fixture row is invalid")
    first_native = first.get("native_output")
    if not isinstance(first_native, Mapping):
        raise ExternalLeagueError("Kaya conformance source witness is missing")
    api = first_native.get("api")
    if not isinstance(api, Mapping):
        raise ExternalLeagueError("Kaya conformance source API witness is missing")
    analysis_api = api.get("GuidelineProcess.analyse_file")
    if not isinstance(analysis_api, Mapping):
        raise ExternalLeagueError("Kaya conformance source path witness is missing")
    module_path = Path(str(analysis_api.get("module_path", ""))).resolve()
    try:
        checkout = module_path.parents[2]
    except IndexError as exc:
        raise ExternalLeagueError("Kaya conformance source path witness is invalid") from exc
    upstream = stored.get("upstream")
    if not isinstance(upstream, Mapping) or _audit_kaya_source_checkout(checkout) != dict(upstream):
        raise ExternalLeagueError("Kaya conformance source provenance no longer verifies")
    environment = stored.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != {
        "root", "python", "python_sha256", "required_distributions",
    }:
        raise ExternalLeagueError("Kaya conformance environment evidence is invalid")
    environment_root = Path(str(environment.get("root", ""))).resolve()
    python = Path(str(environment.get("python", "")))
    resolved_python = python.resolve()
    if (
        not python.is_file() or not resolved_python.is_file()
        or environment.get("python_sha256") != _sha256_file(resolved_python)
        or environment.get("python_sha256") != KAYA_PYTHON_SHA256
        or environment.get("required_distributions") != dict(KAYA_REQUIRED_DISTRIBUTIONS)
        or python.parent.parent.resolve() != environment_root
    ):
        raise ExternalLeagueError("Kaya conformance isolated Python no longer verifies")
    replay_python: Path | None = None
    replay_environment_root: Path | None = None
    if fresh_replay:
        if replay_python_executable is None:
            raise ExternalLeagueError(
                "Kaya fresh replay requires a separately built isolated Python environment"
            )
        replay_python = Path(replay_python_executable).absolute()
        replay_environment_root = replay_python.parent.parent.resolve()
        if (
            not replay_python.is_file()
            or not replay_python.resolve().is_file()
            or _sha256_file(replay_python.resolve()) != KAYA_PYTHON_SHA256
            or replay_environment_root == environment_root
        ):
            raise ExternalLeagueError(
                "Kaya fresh replay environment is missing, unfrozen, or not independent"
            )
    verified_rows: dict[str, dict[str, object]] = {}
    replay_storage_evidence: dict[str, object] | None = None
    for fixture_id in KAYA_REQUIRED_FIXTURE_IDS:
        row = fixture_rows[fixture_id]
        if not isinstance(row, Mapping) or set(row) != {
            "fixture_id", "classification", "input", "direct_input",
            "native_process", "direct_process", "native_output", "direct_output",
        }:
            raise ExternalLeagueError(f"Kaya receipt fixture row is invalid: {fixture_id}")
        if (
            row.get("fixture_id") != fixture_id
            or row.get("classification") != "CONTROLLED_CONFORMANCE_NOT_NATURAL_NOT_SCORING"
        ):
            raise ExternalLeagueError(f"Kaya receipt fixture identity is invalid: {fixture_id}")
        input_ref = row.get("input")
        direct_ref = row.get("direct_input")
        if not isinstance(input_ref, Mapping) or set(input_ref) != {
            "video", "video_sha256", "conversion_receipt", "conversion_receipt_sha256",
            "frame_count", "fps", "shape", "frame_map_sha256",
        } or not isinstance(direct_ref, Mapping) or set(direct_ref) != {"path", "sha256"}:
            raise ExternalLeagueError(f"Kaya receipt fixture input binding is invalid: {fixture_id}")
        video = Path(str(input_ref.get("video", ""))).resolve()
        conversion = Path(str(input_ref.get("conversion_receipt", ""))).resolve()
        direct_manifest = Path(str(direct_ref.get("path", ""))).resolve()
        try:
            direct_manifest.relative_to(path.parent)
        except ValueError as exc:
            raise ExternalLeagueError(f"Kaya direct input escapes receipt ownership: {fixture_id}") from exc
        if (
            not video.is_file() or input_ref.get("video_sha256") != _sha256_file(video)
            or not conversion.is_file() or input_ref.get("conversion_receipt_sha256") != _sha256_file(conversion)
            or not direct_manifest.is_file() or direct_ref.get("sha256") != _sha256_file(direct_manifest)
        ):
            raise ExternalLeagueError(f"Kaya receipt fixture artifact hash mismatches: {fixture_id}")
        _, contract = _canonical_decoder_timeline_contract(video, conversion)
        if any(input_ref.get(field) != contract[field] for field in ("frame_count", "fps", "shape", "frame_map_sha256")):
            raise ExternalLeagueError(f"Kaya receipt canonical input contract drifted: {fixture_id}")
        native_process = row.get("native_process")
        direct_process = row.get("direct_process")
        if not isinstance(native_process, Mapping) or not isinstance(direct_process, Mapping):
            raise ExternalLeagueError(f"Kaya receipt process evidence is invalid: {fixture_id}")
        for mode, process in (("native", native_process), ("direct", direct_process)):
            process_fields = {
                "command", "command_sha256", "environment", "environment_sha256",
                "working_directory", "started_monotonic_ns", "finished_monotonic_ns",
                "wall_time_ns", "timeout_seconds", "timed_out", "exit_code",
                "stdout", "stderr", "output",
            }
            if set(process) != process_fields:
                raise ExternalLeagueError(f"Kaya receipt process fields are invalid: {fixture_id}:{mode}")
            process_root = Path(str(process.get("working_directory", ""))).resolve()
            try:
                process_root.relative_to(path.parent)
            except ValueError as exc:
                raise ExternalLeagueError(f"Kaya receipt process root escapes receipt ownership: {fixture_id}:{mode}") from exc
            if process_root != path.parent / fixture_id / mode:
                raise ExternalLeagueError(f"Kaya receipt process root is not canonical: {fixture_id}:{mode}")
            direct_reuse_count_for_command = 2 if fixture_id == "state-reuse" else 1
            reuse_count_for_command = 1 if mode == "native" else direct_reuse_count_for_command
            output_path = process_root / "child-output.json"
            expected_command = [
                str(python), "-S", "-X", f"pycache_prefix={process_root / 'pycache'}", "-c",
                _KAYA_CONFORMANCE_CHILD_SCRIPT, mode, str(checkout), str(video),
                str(direct_manifest) if mode == "direct" else "-", str(output_path),
                _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8")),
                str(reuse_count_for_command),
            ]
            expected_environment = _kaya_child_environment(environment_root, process_root / "home")
            started = process.get("started_monotonic_ns")
            finished = process.get("finished_monotonic_ns")
            wall_time = process.get("wall_time_ns")
            timeout = process.get("timeout_seconds")
            if (
                process.get("exit_code") != 0 or process.get("timed_out") is not False
                or process.get("command") != expected_command
                or process.get("command_sha256") != _canonical_json_sha256(process.get("command"))
                or process.get("environment") != expected_environment
                or process.get("environment_sha256") != _canonical_json_sha256(process.get("environment"))
                or isinstance(started, bool) or not isinstance(started, int)
                or isinstance(finished, bool) or not isinstance(finished, int) or finished < started
                or isinstance(wall_time, bool) or not isinstance(wall_time, int) or wall_time != finished - started
                or isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
            ):
                raise ExternalLeagueError(f"Kaya receipt process did not complete cleanly: {fixture_id}:{mode}")
            output_ref = process.get("output")
            if (
                not isinstance(output_ref, Mapping)
                or output_ref.get("path") != str(output_path)
                or output_ref.get("exists") is not True
            ):
                raise ExternalLeagueError(f"Kaya receipt process output path is invalid: {fixture_id}:{mode}")
            for stream in ("stdout", "stderr"):
                stream_ref = process.get(stream)
                if not isinstance(stream_ref, Mapping):
                    raise ExternalLeagueError(f"Kaya receipt stream evidence is invalid: {fixture_id}:{mode}:{stream}")
                stream_path = Path(str(stream_ref.get("path", ""))).resolve()
                if (
                    stream_path != process_root / f"{stream}.bin"
                    or not stream_path.is_file()
                    or stream_ref.get("sha256") != _sha256_file(stream_path)
                ):
                    raise ExternalLeagueError(f"Kaya receipt stream hash mismatches: {fixture_id}:{mode}:{stream}")
        native_output = _validate_kaya_child_output(
            _load_kaya_child_output(native_process), mode="native", reuse_count=1,
            checkout=checkout, environment_root=environment_root,
            video=video, conversion=conversion, direct_manifest=None,
        )
        direct_reuse_count = 2 if fixture_id == "state-reuse" else 1
        direct_output = _validate_kaya_child_output(
            _load_kaya_child_output(direct_process), mode="direct", reuse_count=direct_reuse_count,
            checkout=checkout, environment_root=environment_root,
            video=video, conversion=conversion, direct_manifest=direct_manifest,
        )
        if native_output != row.get("native_output") or direct_output != row.get("direct_output"):
            raise ExternalLeagueError(f"Kaya receipt embedded output diverges from raw child artifact: {fixture_id}")
        if fresh_replay:
            assert replay_python is not None
            assert replay_environment_root is not None
            scratch_root = Path(os.environ.get("FLASHPATCH_KAYA_REPLAY_SCRATCH", path.parent)).resolve()
            if scratch_root != path.parent:
                scratch_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"kaya-replay-{fixture_id}-", dir=scratch_root) as temporary:
                replay_root = Path(temporary)
                native_replay = _execute_kaya_child(
                    mode="native", checkout=checkout, python=replay_python, video=video,
                    direct_manifest=None, output_root=replay_root / "native", reuse_count=1,
                    timeout_seconds=int(native_process["timeout_seconds"]),
                )
                direct_replay = _execute_kaya_child(
                    mode="direct", checkout=checkout, python=replay_python, video=video,
                    direct_manifest=direct_manifest, output_root=replay_root / "direct",
                    reuse_count=direct_reuse_count,
                    timeout_seconds=int(direct_process["timeout_seconds"]),
                )
                replay_native_output = _validate_kaya_child_output(
                    _load_kaya_child_output(native_replay), mode="native", reuse_count=1,
                    checkout=checkout, environment_root=replay_environment_root,
                    video=video, conversion=conversion, direct_manifest=None,
                )
                replay_direct_output = _validate_kaya_child_output(
                    _load_kaya_child_output(direct_replay), mode="direct", reuse_count=direct_reuse_count,
                    checkout=checkout, environment_root=replay_environment_root,
                    video=video, conversion=conversion, direct_manifest=direct_manifest,
                )
                if replay_storage_evidence is None:
                    replay_storage_evidence = _verify_kaya_replay_storage_independence(
                        native_output,
                        replay_native_output,
                        primary_environment_root=environment_root,
                        replay_environment_root=replay_environment_root,
                    )
                if (
                    _kaya_cross_environment_projection(
                        replay_native_output, environment_root=replay_environment_root
                    )
                    != _kaya_cross_environment_projection(
                        native_output, environment_root=environment_root
                    )
                    or _kaya_cross_environment_projection(
                        replay_direct_output, environment_root=replay_environment_root
                    )
                    != _kaya_cross_environment_projection(
                        direct_output, environment_root=environment_root
                    )
                ):
                    raise ExternalLeagueError(f"Kaya fresh replay differs from stored child output: {fixture_id}")
        verified_rows[fixture_id] = {**dict(row), "native_output": native_output, "direct_output": direct_output}
    failures = _kaya_semantic_conformance_failures(verified_rows)
    expected_conformance = {
        "native_api": "unmodified_GuidelineProcess.analyse_file",
        "direct_api": "same_unmodified_GuidelineProcess.analyse_file_with_receipt_bound_DirectCapture",
        "rgb_to_bgr": "exactly_one_channel_reverse_before_upstream_consumption",
        "result_equality": "exact_raw_arrays_and_exact_upstream_interval_tuples_no_normalization",
        "eof_open_both_bug": "preserved_unmodified",
        "failures": failures,
    }
    if stored.get("conformance") != expected_conformance:
        raise ExternalLeagueError("Kaya receipt semantic conclusion drifted")
    expected_status = "VERIFIED" if not failures else "NOT_VERIFIED"
    if stored.get("status") != "NOT_VERIFIED":
        raise ExternalLeagueError("Kaya pre-replay receipt cannot claim VERIFIED status")
    result = _finalize_kaya_verification(
        stored,
        expected_status=expected_status,
        failures=failures,
        fresh_replay=fresh_replay,
    )
    if fresh_replay:
        if replay_environment_root is None or replay_python is None or replay_storage_evidence is None:
            raise ExternalLeagueError("Kaya fresh replay independence evidence is missing")
        result["replay_environment"] = {
            "root": str(replay_environment_root),
            "python": str(replay_python),
            "python_sha256": _sha256_file(replay_python.resolve()),
        }
        result["dependency_storage_independence"] = replay_storage_evidence
    return result


_KAYA_PARTICIPANT_SCOREABLE_BLOCKERS = [
    "natural_public_case_ledger_missing",
    "independent_gold_receipts_missing",
    "same_input_decode_parity_receipts_missing",
    "equal_budget_three_repeat_receipts_missing",
]


def write_kaya_participant_conformance_receipt(
    prototype_receipt: Path | str,
    replay_python_executable: Path | str,
    receipt_path: Path | str,
) -> dict[str, object]:
    """Promote verified Kaya conformance to an unscored L7 identity.

    The historical prototype receipt remains immutable and keeps its prototype
    identity.  This derivative receipt binds a successful independent fresh
    replay to the canonical participant identity, but grants no score, natural
    corpus, gold, fair-runtime, blind-league, or external-claim authority.
    """
    source = Path(prototype_receipt).resolve()
    destination = Path(receipt_path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Kaya participant conformance receipt already exists: {destination}"
        )
    verified = verify_kaya_conformance_prototype(
        source,
        fresh_replay=True,
        replay_python_executable=replay_python_executable,
    )
    if (
        verified.get("verification_status") != "VERIFIED"
        or verified.get("fresh_replay_verified") is not True
        or verified.get("scoreable") is not False
        or verified.get("identity") != KAYA_PROTOTYPE_ID
        or verified.get("conformance", {}).get("failures") != []
    ):
        raise ExternalLeagueError(
            "Kaya prototype has not passed exact independent conformance replay"
        )
    primary_environment = verified.get("environment")
    replay_environment = verified.get("replay_environment")
    storage = verified.get("dependency_storage_independence")
    if (
        not isinstance(primary_environment, Mapping)
        or not isinstance(replay_environment, Mapping)
        or not isinstance(storage, Mapping)
        or storage.get("shared_inode_count") != 0
    ):
        raise ExternalLeagueError(
            "Kaya participant promotion lacks independent runtime storage evidence"
        )
    receipt: dict[str, object] = {
        "schema": KAYA_PARTICIPANT_CONFORMANCE_SCHEMA,
        "identity": KAYA_DIRECT_PARTICIPANT_ID,
        "prototype_identity": KAYA_PROTOTYPE_ID,
        "classification": "UNSCORED_DIRECT_PARTICIPANT_CONFORMANCE",
        "upstream": {
            "repository_url": KAYA_REPOSITORY_URL,
            "revision": KAYA_SOURCE_REVISION,
            "tree": KAYA_SOURCE_TREE,
        },
        "prototype_receipt": {
            "path": str(source),
            "sha256": _sha256_file(source),
        },
        "verification": {
            "status": "VERIFIED",
            "fresh_replay_verified": True,
            "fixture_ids": list(KAYA_REQUIRED_FIXTURE_IDS),
            "conformance_sha256": _canonical_json_sha256(verified["conformance"]),
            "primary_environment": dict(primary_environment),
            "replay_environment": dict(replay_environment),
            "dependency_storage_independence": dict(storage),
        },
        "status": "VERIFIED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "unscored_population_authorized": True,
        "external_claim_authorized": False,
        "scoreable_blockers": list(_KAYA_PARTICIPANT_SCOREABLE_BLOCKERS),
    }
    _write_json(destination, receipt)
    return {**receipt, "receipt": str(destination)}


def verify_kaya_participant_conformance_receipt(
    receipt_path: Path | str,
) -> dict[str, object]:
    """Reopen a Kaya promotion receipt without granting scoring authority."""
    path = Path(receipt_path).resolve()
    cache_key = str(path)
    cached = _KAYA_PARTICIPANT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError(
            "Kaya participant conformance receipt is unreadable"
        ) from exc
    expected_fields = {
        "schema",
        "identity",
        "prototype_identity",
        "classification",
        "upstream",
        "prototype_receipt",
        "verification",
        "status",
        "claim_status",
        "scoreable",
        "unscored_population_authorized",
        "external_claim_authorized",
        "scoreable_blockers",
    }
    if (
        not isinstance(stored, Mapping)
        or set(stored) != expected_fields
        or stored.get("schema") != KAYA_PARTICIPANT_CONFORMANCE_SCHEMA
        or stored.get("identity") != KAYA_DIRECT_PARTICIPANT_ID
        or stored.get("prototype_identity") != KAYA_PROTOTYPE_ID
        or stored.get("classification")
        != "UNSCORED_DIRECT_PARTICIPANT_CONFORMANCE"
        or stored.get("status") != "VERIFIED"
        or stored.get("claim_status") != "NOT_SCOREABLE"
        or stored.get("scoreable") is not False
        or stored.get("unscored_population_authorized") is not True
        or stored.get("external_claim_authorized") is not False
        or stored.get("scoreable_blockers")
        != _KAYA_PARTICIPANT_SCOREABLE_BLOCKERS
        or stored.get("upstream")
        != {
            "repository_url": KAYA_REPOSITORY_URL,
            "revision": KAYA_SOURCE_REVISION,
            "tree": KAYA_SOURCE_TREE,
        }
    ):
        raise ExternalLeagueError(
            "Kaya participant conformance identity or claim boundary is invalid"
        )
    prototype_ref = stored.get("prototype_receipt")
    if (
        not isinstance(prototype_ref, Mapping)
        or set(prototype_ref) != {"path", "sha256"}
        or not isinstance(prototype_ref.get("path"), str)
        or not isinstance(prototype_ref.get("sha256"), str)
    ):
        raise ExternalLeagueError(
            "Kaya participant conformance prototype binding is invalid"
        )
    prototype = Path(str(prototype_ref["path"])).resolve()
    if (
        not prototype.is_file()
        or prototype_ref["sha256"] != _sha256_file(prototype)
    ):
        raise ExternalLeagueError(
            "Kaya participant conformance prototype receipt hash mismatches"
        )
    verification = stored.get("verification")
    if not isinstance(verification, Mapping) or set(verification) != {
        "status",
        "fresh_replay_verified",
        "fixture_ids",
        "conformance_sha256",
        "primary_environment",
        "replay_environment",
        "dependency_storage_independence",
    }:
        raise ExternalLeagueError(
            "Kaya participant conformance verification evidence is invalid"
        )
    primary = verification.get("primary_environment")
    replay = verification.get("replay_environment")
    storage = verification.get("dependency_storage_independence")
    if (
        not isinstance(replay, Mapping)
        or set(replay) != {"root", "python", "python_sha256"}
        or not isinstance(replay.get("python"), str)
    ):
        raise ExternalLeagueError(
            "Kaya participant replay environment binding is invalid"
        )
    reverified = verify_kaya_conformance_prototype(
        prototype,
        fresh_replay=True,
        replay_python_executable=str(replay["python"]),
    )
    if (
        verification.get("status") != "VERIFIED"
        or verification.get("fresh_replay_verified") is not True
        or verification.get("fixture_ids") != list(KAYA_REQUIRED_FIXTURE_IDS)
        or verification.get("conformance_sha256")
        != _canonical_json_sha256(reverified["conformance"])
        or reverified.get("conformance", {}).get("failures") != []
        or reverified.get("verification_status") != "VERIFIED"
        or reverified.get("fresh_replay_verified") is not True
        or primary != reverified.get("environment")
        or not isinstance(primary, Mapping)
        or replay != reverified.get("replay_environment")
        or storage != reverified.get("dependency_storage_independence")
        or replay.get("python_sha256") != KAYA_PYTHON_SHA256
        or replay.get("root") == primary.get("root")
        or not isinstance(storage, Mapping)
        or set(storage)
        != {
            "primary_identity_count",
            "replay_identity_count",
            "primary_counts",
            "replay_counts",
            "shared_inode_count",
        }
        or storage.get("shared_inode_count") != 0
        or not isinstance(storage.get("primary_identity_count"), int)
        or not isinstance(storage.get("replay_identity_count"), int)
        or int(storage["primary_identity_count"]) <= 0
        or int(storage["replay_identity_count"]) <= 0
    ):
        raise ExternalLeagueError(
        "Kaya participant conformance verification is not exact and independent"
        )
    result = dict(stored)
    _KAYA_PARTICIPANT_CACHE[cache_key] = result
    return result


def _kaya_natural_case_exact_failures(
    native_output: Mapping[str, object],
    direct_output: Mapping[str, object],
) -> list[str]:
    """Return the non-negotiable evidence mismatches for one canonical case."""
    failures: list[str] = []
    native_captures = native_output.get("capture_runs")
    direct_captures = direct_output.get("capture_runs")
    if (
        not isinstance(native_captures, list)
        or not isinstance(direct_captures, list)
        or len(native_captures) != 1
        or len(direct_captures) != 1
        or native_captures[0].get("ledger") != direct_captures[0].get("ledger")
    ):
        failures.append("native_direct_preconsumption_bgr_ledger_mismatch")
    if native_output.get("results") != direct_output.get("results"):
        failures.append("native_direct_raw_arrays_or_intervals_mismatch")
    return failures


def _validate_kaya_natural_process(
    process: Mapping[str, object],
    *,
    receipt_root: Path,
    mode: str,
) -> None:
    fields = {
        "command", "command_sha256", "environment", "environment_sha256",
        "working_directory", "started_monotonic_ns", "finished_monotonic_ns",
        "wall_time_ns", "timeout_seconds", "timed_out", "exit_code",
        "stdout", "stderr", "output",
    }
    if set(process) != fields:
        raise ExternalLeagueError(f"Kaya natural case process fields are invalid: {mode}")
    root = Path(str(process.get("working_directory", ""))).resolve()
    if root != receipt_root / mode:
        raise ExternalLeagueError(f"Kaya natural case process root is invalid: {mode}")
    started = process.get("started_monotonic_ns")
    finished = process.get("finished_monotonic_ns")
    wall_time = process.get("wall_time_ns")
    if (
        process.get("exit_code") != 0
        or process.get("timed_out") is not False
        or process.get("command_sha256") != _canonical_json_sha256(process.get("command"))
        or process.get("environment_sha256") != _canonical_json_sha256(process.get("environment"))
        or isinstance(started, bool) or not isinstance(started, int)
        or isinstance(finished, bool) or not isinstance(finished, int) or finished < started
        or isinstance(wall_time, bool) or not isinstance(wall_time, int) or wall_time != finished - started
        or isinstance(process.get("timeout_seconds"), bool)
        or not isinstance(process.get("timeout_seconds"), int)
        or int(process["timeout_seconds"]) <= 0
    ):
        raise ExternalLeagueError(f"Kaya natural case process did not complete cleanly: {mode}")
    for label, filename in (("stdout", "stdout.bin"), ("stderr", "stderr.bin")):
        reference = process.get(label)
        if not isinstance(reference, Mapping):
            raise ExternalLeagueError(f"Kaya natural case stream reference is invalid: {mode}:{label}")
        path = Path(str(reference.get("path", ""))).resolve()
        if path != root / filename or not path.is_file() or reference.get("sha256") != _sha256_file(path):
            raise ExternalLeagueError(f"Kaya natural case stream hash mismatches: {mode}:{label}")
    output = process.get("output")
    if not isinstance(output, Mapping):
        raise ExternalLeagueError(f"Kaya natural case output reference is invalid: {mode}")
    output_path = Path(str(output.get("path", ""))).resolve()
    if (
        output_path != root / "child-output.json"
        or output.get("exists") is not True
        or not output_path.is_file()
        or output.get("sha256") != _sha256_file(output_path)
    ):
        raise ExternalLeagueError(f"Kaya natural case child output hash mismatches: {mode}")


def execute_kaya_natural_case_parity(
    participant_conformance_receipt: Path | str,
    *,
    checkout: Path | str,
    python_executable: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    """Run native and direct Kaya on one canonical FFV1 input without scoring it."""
    participant_path = Path(participant_conformance_receipt).resolve()
    verify_kaya_participant_conformance_receipt(participant_path)
    project = Path(checkout).resolve()
    # Keep the venv entrypoint lexical.  ``env/bin/python`` commonly links to
    # the base interpreter; resolving it before the child is launched silently
    # discards the environment's site-packages.  Hash the resolved executable,
    # but execute through the declared isolated environment path.
    python = Path(python_executable).absolute()
    resolved_python = python.resolve()
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Kaya natural case output root already exists: {root}")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ExternalLeagueError("Kaya natural case timeout must be a positive integer")
    provenance = _audit_kaya_source_checkout(project)
    if (
        not python.is_file()
        or python.parent.name != "bin"
        or not resolved_python.is_file()
        or _sha256_file(resolved_python) != KAYA_PYTHON_SHA256
    ):
        raise ExternalLeagueError("Kaya natural case Python is not the frozen interpreter")
    _, contract = _canonical_decoder_timeline_contract(video, conversion)
    root.mkdir(parents=True)
    direct = _materialize_kaya_direct_input("natural-case", video, conversion, root / "direct-input")
    manifest = Path(str(direct["manifest"])).resolve()
    native_process = _execute_kaya_child(
        mode="native", checkout=project, python=python, video=video,
        direct_manifest=None, output_root=root / "native", reuse_count=1,
        timeout_seconds=timeout_seconds,
    )
    direct_process = _execute_kaya_child(
        mode="direct", checkout=project, python=python, video=video,
        direct_manifest=manifest, output_root=root / "direct", reuse_count=1,
        timeout_seconds=timeout_seconds,
    )
    environment_root = python.parent.parent.resolve()
    native_output = _validate_kaya_child_output(
        _load_kaya_child_output(native_process), mode="native", reuse_count=1,
        checkout=project, environment_root=environment_root, video=video,
        conversion=conversion, direct_manifest=None,
    )
    direct_output = _validate_kaya_child_output(
        _load_kaya_child_output(direct_process), mode="direct", reuse_count=1,
        checkout=project, environment_root=environment_root, video=video,
        conversion=conversion, direct_manifest=manifest,
    )
    failures = _kaya_natural_case_exact_failures(native_output, direct_output)
    receipt: dict[str, object] = {
        "schema": KAYA_NATURAL_CASE_PARITY_SCHEMA,
        "identity": KAYA_DIRECT_PARTICIPANT_ID,
        "prototype_identity": KAYA_PROTOTYPE_ID,
        "classification": "NATURAL_CASE_INPUT_PARITY_NOT_SCORING",
        "participant_conformance": {"path": str(participant_path), "sha256": _sha256_file(participant_path)},
        "source_checkout": str(project),
        "upstream": provenance,
        "runtime": {"python": str(python), "python_sha256": _sha256_file(resolved_python)},
        "canonical_contract": contract,
        "direct_input": {"path": str(manifest), "sha256": _sha256_file(manifest)},
        "native_process": native_process,
        "direct_process": direct_process,
        "native_output": native_output,
        "direct_output": direct_output,
        "parity_failures": failures,
        "status": "VERIFIED" if not failures else "NOT_VERIFIED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "natural_case_parity_is_not_an_independent_execution_witness",
            "independent_gold_receipts_missing",
            "equal_budget_three_repeat_receipts_missing",
        ],
    }
    receipt_path = root / "kaya-natural-case-parity-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def verify_kaya_natural_case_parity_receipt(
    receipt_path: Path | str,
) -> dict[str, object]:
    """Reopen one non-scoreable Kaya natural-case parity receipt fail-closed."""
    path = Path(receipt_path).resolve()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("Kaya natural case parity receipt is unreadable") from exc
    fields = {
        "schema", "identity", "prototype_identity", "classification",
        "participant_conformance", "source_checkout", "upstream", "runtime", "canonical_contract",
        "direct_input", "native_process", "direct_process", "native_output",
        "direct_output", "parity_failures", "status", "claim_status", "scoreable",
        "comparison_eligible", "claim_blockers",
    }
    if (
        not isinstance(stored, Mapping) or set(stored) != fields
        or stored.get("schema") != KAYA_NATURAL_CASE_PARITY_SCHEMA
        or stored.get("identity") != KAYA_DIRECT_PARTICIPANT_ID
        or stored.get("prototype_identity") != KAYA_PROTOTYPE_ID
        or stored.get("classification") != "NATURAL_CASE_INPUT_PARITY_NOT_SCORING"
        or stored.get("claim_status") != "NOT_SCOREABLE"
        or stored.get("scoreable") is not False
        or stored.get("comparison_eligible") is not False
        or stored.get("claim_blockers") != [
            "natural_case_parity_is_not_an_independent_execution_witness",
            "independent_gold_receipts_missing",
            "equal_budget_three_repeat_receipts_missing",
        ]
    ):
        raise ExternalLeagueError("Kaya natural case parity identity or claim boundary is invalid")
    participant_ref = stored.get("participant_conformance")
    if not isinstance(participant_ref, Mapping) or set(participant_ref) != {"path", "sha256"}:
        raise ExternalLeagueError("Kaya natural case participant conformance binding is invalid")
    participant_path = Path(str(participant_ref.get("path", ""))).resolve()
    if not participant_path.is_file() or participant_ref.get("sha256") != _sha256_file(participant_path):
        raise ExternalLeagueError("Kaya natural case participant conformance hash mismatches")
    verify_kaya_participant_conformance_receipt(participant_path)
    runtime = stored.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {"python", "python_sha256"}:
        raise ExternalLeagueError("Kaya natural case runtime binding is invalid")
    python = Path(str(runtime.get("python", ""))).absolute()
    resolved_python = python.resolve()
    if (
        not python.is_file()
        or python.parent.name != "bin"
        or not resolved_python.is_file()
        or runtime.get("python_sha256") != _sha256_file(resolved_python)
        or _sha256_file(resolved_python) != KAYA_PYTHON_SHA256
    ):
        raise ExternalLeagueError("Kaya natural case frozen Python hash mismatches")
    upstream = stored.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ExternalLeagueError("Kaya natural case source provenance is invalid")
    checkout = Path(str(stored.get("source_checkout", ""))).resolve()
    if _audit_kaya_source_checkout(checkout) != dict(upstream):
        raise ExternalLeagueError("Kaya natural case source provenance drifted")
    contract = stored.get("canonical_contract")
    if not isinstance(contract, Mapping):
        raise ExternalLeagueError("Kaya natural case canonical contract is invalid")
    video = Path(str(contract.get("canonical_video", {}).get("path", ""))).resolve()
    conversion = Path(str(contract.get("conversion_receipt", {}).get("path", ""))).resolve()
    _, expected_contract = _canonical_decoder_timeline_contract(video, conversion)
    if dict(contract) != expected_contract:
        raise ExternalLeagueError("Kaya natural case canonical contract drifted")
    direct_ref = stored.get("direct_input")
    if not isinstance(direct_ref, Mapping) or set(direct_ref) != {"path", "sha256"}:
        raise ExternalLeagueError("Kaya natural case direct input binding is invalid")
    direct_manifest = Path(str(direct_ref.get("path", ""))).resolve()
    try:
        direct_manifest.relative_to(path.parent)
    except ValueError as exc:
        raise ExternalLeagueError("Kaya natural case direct input escapes receipt ownership") from exc
    if not direct_manifest.is_file() or direct_ref.get("sha256") != _sha256_file(direct_manifest):
        raise ExternalLeagueError("Kaya natural case direct input hash mismatches")
    _load_kaya_direct_input_manifest(direct_manifest, expected_video=video, expected_conversion=conversion)
    processes = (("native", stored.get("native_process"), None), ("direct", stored.get("direct_process"), direct_manifest))
    outputs: dict[str, dict[str, object]] = {}
    for mode, process, direct in processes:
        if not isinstance(process, Mapping):
            raise ExternalLeagueError(f"Kaya natural case process is invalid: {mode}")
        _validate_kaya_natural_process(process, receipt_root=path.parent, mode=mode)
        process_root = path.parent / mode
        expected_command = [
            str(python), "-S", "-X", f"pycache_prefix={process_root / 'pycache'}", "-c",
            _KAYA_CONFORMANCE_CHILD_SCRIPT, mode, str(checkout), str(video),
            str(direct) if direct is not None else "-", str(process_root / "child-output.json"),
            _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8")), "1",
        ]
        if process.get("command") != expected_command:
            raise ExternalLeagueError(f"Kaya natural case child command drifted: {mode}")
        output = _validate_kaya_child_output(
            _load_kaya_child_output(process), mode=mode, reuse_count=1,
            checkout=checkout, environment_root=python.parent.parent.resolve(), video=video,
            conversion=conversion, direct_manifest=direct,
        )
        if output != stored.get(f"{mode}_output"):
            raise ExternalLeagueError(f"Kaya natural case embedded output diverges from child artifact: {mode}")
        outputs[mode] = output
    failures = _kaya_natural_case_exact_failures(outputs["native"], outputs["direct"])
    if stored.get("parity_failures") != failures or stored.get("status") != ("VERIFIED" if not failures else "NOT_VERIFIED"):
        raise ExternalLeagueError("Kaya natural case parity conclusion drifted")
    return dict(stored)


def _kaya_fair_runtime_prerequisites(
    participant_conformance_receipt: Path | str,
    natural_case_parity_receipt: Path | str,
    *,
    checkout: Path,
    python: Path,
    canonical_video: Path,
    conversion_receipt: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Reopen Kaya's promotion and exact native/direct input prerequisites."""
    participant_path = Path(participant_conformance_receipt).resolve()
    parity_path = Path(natural_case_parity_receipt).resolve()
    participant = verify_kaya_participant_conformance_receipt(participant_path)
    parity = verify_kaya_natural_case_parity_receipt(parity_path)
    participant_ref = parity.get("participant_conformance")
    contract = parity.get("canonical_contract")
    runtime = parity.get("runtime")
    if (
        participant.get("status") != "VERIFIED"
        or participant.get("scoreable") is not False
        or participant.get("external_claim_authorized") is not False
        or parity.get("status") != "VERIFIED"
        or parity.get("parity_failures") != []
        or parity.get("claim_status") != "NOT_SCOREABLE"
        or parity.get("scoreable") is not False
        or parity.get("comparison_eligible") is not False
        or not isinstance(participant_ref, Mapping)
        or participant_ref != {
            "path": str(participant_path),
            "sha256": _sha256_file(participant_path),
        }
        or not isinstance(contract, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        raise ExternalLeagueError("Kaya fair-runtime prerequisites are not verified")
    canonical_ref = contract.get("canonical_video")
    conversion_ref = contract.get("conversion_receipt")
    if (
        not isinstance(canonical_ref, Mapping)
        or not isinstance(conversion_ref, Mapping)
        or canonical_ref != {
            "path": str(canonical_video),
            "sha256": _sha256_file(canonical_video),
        }
        or conversion_ref != {
            "path": str(conversion_receipt),
            "sha256": _sha256_file(conversion_receipt),
        }
        or Path(str(parity.get("source_checkout", ""))).resolve() != checkout
        or Path(str(runtime.get("python", ""))).absolute() != python
        or runtime.get("python_sha256") != _sha256_file(python.resolve())
    ):
        raise ExternalLeagueError("Kaya fair-runtime input, source, or runtime differs from natural parity")
    return participant_path, parity_path, parity


def _normalize_kaya_fair_runtime_output(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Project exact upstream arrays into a stable terminal detector observation."""
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise ExternalLeagueError("Kaya fair-runtime output must contain one upstream result")
    result = results[0]
    raw = result.get("raw")
    intervals = result.get("interval_tuples")
    if not isinstance(raw, Mapping) or not isinstance(intervals, list):
        raise ExternalLeagueError("Kaya fair-runtime raw result is invalid")
    general = raw.get("General Flashes")
    red = raw.get("Red Flashes")
    if not isinstance(general, list) or not isinstance(red, list) or len(general) != len(red):
        raise ExternalLeagueError("Kaya fair-runtime raw arrays are invalid")
    hazardous_indices: set[int] = set()
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 3:
            raise ExternalLeagueError("Kaya fair-runtime interval tuple is invalid")
        hazardous_indices.update(range(int(interval[1]), int(interval[2])))
    return {
        "schema": "flashpatch-l7-kaya-normalized-observation-v1",
        "prediction": "HAZARDOUS" if intervals else "SAFE",
        "hazard_frame_indices": sorted(hazardous_indices),
        "interval_tuples": intervals,
        "frame_count": len(general),
        "raw_results_sha256": _canonical_json_sha256(results),
    }


def execute_kaya_scheduled_fair_runtime(
    participant_conformance_receipt: Path | str,
    natural_case_parity_receipt: Path | str,
    *,
    checkout: Path | str,
    python_executable: Path | str,
    canonical_video: Path | str,
    conversion_receipt: Path | str,
    output_root: Path | str,
    runtime_protocol: FairRuntimeProtocol | Mapping[str, object],
    scheduled_repeat_ordinal: int,
    runtime_schedule: Path | str,
    schedule_slot: int,
) -> dict[str, object]:
    """Run Kaya's exact native FFV1 path in one pre-frozen fair-runtime slot."""
    project = Path(checkout).resolve()
    python = Path(python_executable).absolute()
    resolved_python = python.resolve()
    video = Path(canonical_video).resolve()
    conversion = Path(conversion_receipt).resolve()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Kaya fair-runtime output root already exists: {root}")
    frozen = _freeze_runtime_protocol_input(runtime_protocol)
    if frozen is None:
        raise ExternalLeagueError("Kaya fair-runtime requires a frozen protocol")
    if (
        not video.is_file()
        or not conversion.is_file()
        or not python.is_file()
        or python.parent.name != "bin"
        or not resolved_python.is_file()
        or _sha256_file(resolved_python) != KAYA_PYTHON_SHA256
    ):
        raise ExternalLeagueError("Kaya fair-runtime input or frozen Python is unavailable")
    provenance = _audit_kaya_source_checkout(project)
    participant_path, parity_path, _ = _kaya_fair_runtime_prerequisites(
        participant_conformance_receipt,
        natural_case_parity_receipt,
        checkout=project,
        python=python,
        canonical_video=video,
        conversion_receipt=conversion,
    )
    schedule_binding = _load_schedule_assignment(
        runtime_schedule,
        schedule_slot=schedule_slot,
        protocol=frozen,
        comparator=KAYA_DIRECT_PARTICIPANT_ID,
        repeat_ordinal=scheduled_repeat_ordinal,
        input_sha256=_sha256_file(video),
    )
    if schedule_binding is None:
        raise ExternalLeagueError("Kaya fair-runtime requires a pre-frozen schedule slot")

    root.mkdir(parents=True)
    raw_output = root / "child-output.json"
    stdout_path = root / "stdout.bin"
    stderr_path = root / "stderr.bin"
    probe_path = root / "runtime-probe.json"
    adapter_hash = _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
    tool_command = [
        str(python), "-S", "-X", f"pycache_prefix={root / 'pycache'}", "-c",
        _KAYA_CONFORMANCE_CHILD_SCRIPT, "native", str(project), str(video), "-",
        str(raw_output), adapter_hash, "1",
    ]
    normalized_observation: dict[str, object] | None = None
    validated_output: dict[str, object] | None = None
    parse_error: str | None = None
    child_probe: dict[str, object] | None = None
    timed_out = False
    with _fair_execution_context(
        frozen,
        video,
        base_environment=None,
        schedule_binding=schedule_binding,
        launcher_cwd=root,
    ) as execution:
        command = _instrument_fair_command(execution, tool_command, probe_path)
        started = time.monotonic_ns()
        conversion_payload = _load_conversion_receipt(conversion, video)
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                check=False,
                timeout=int(frozen["budget"]["timeout_seconds"]),
                env=execution["environment"],
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"\nflashpatch: Kaya fair-runtime timeout",
            )
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        if completed.returncode == 0 and raw_output.is_file() and raw_output.stat().st_size > 0:
            process_ref = {
                "output": {
                    "path": str(raw_output),
                    "exists": True,
                    "sha256": _sha256_file(raw_output),
                }
            }
            try:
                validated_output = _validate_kaya_child_output(
                    _load_kaya_child_output(process_ref),
                    mode="native",
                    reuse_count=1,
                    checkout=project,
                    environment_root=python.parent.parent.resolve(),
                    video=video,
                    conversion=conversion,
                    direct_manifest=None,
                )
                normalized_observation = _normalize_kaya_fair_runtime_output(validated_output)
            except ExternalLeagueError as exc:
                parse_error = str(exc)
        try:
            child_probe = _load_child_runtime_probe(probe_path)
        except ExternalLeagueError:
            child_probe = None
        finished = time.monotonic_ns()
        elapsed_ns = finished - started
        if elapsed_ns > int(frozen["budget"]["timeout_seconds"]) * 1_000_000_000:
            timed_out = True
        process_valid = (
            completed.returncode == 0
            and raw_output.is_file()
            and raw_output.stat().st_size > 0
            and validated_output is not None
            and normalized_observation is not None
            and child_probe is not None
            and not timed_out
        )
        runtime_receipt = _fair_runtime_run_receipt(
            frozen,
            comparator=KAYA_DIRECT_PARTICIPANT_ID,
            scheduled_repeat_ordinal=scheduled_repeat_ordinal,
            schedule_binding=schedule_binding,
            input_sha256=_sha256_file(video),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            wall_time_ns=elapsed_ns,
            timed_out=timed_out,
            observation=normalized_observation,
            normalizer="kaya-native-exact-v1",
            observed_environment={
                "parent_precondition": execution["observation"],
                "child_probe": child_probe,
            },
        )
    receipt = {
        "schema": KAYA_FAIR_RUNTIME_RUN_SCHEMA,
        "comparator": {
            "name": KAYA_DIRECT_PARTICIPANT_ID,
            "repository_url": KAYA_REPOSITORY_URL,
            "revision": KAYA_SOURCE_REVISION,
            "tree": KAYA_SOURCE_TREE,
            "license": "BSD-3-Clause",
            "binary": str(python),
            "binary_sha256": _sha256_file(resolved_python),
            "working_directory": str(root),
            "source_checkout": str(project),
            "upstream": provenance,
        },
        "input": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {
            "path": str(conversion),
            "sha256": _sha256_file(conversion),
            "renderer_rgb_sha256": conversion_payload["renderer_rgb"]["raw_sha256"],
        },
        "participant_conformance": {
            "path": str(participant_path), "sha256": _sha256_file(participant_path),
        },
        "natural_case_parity": {
            "path": str(parity_path), "sha256": _sha256_file(parity_path),
        },
        "command": command,
        "exit_code": completed.returncode,
        "wall_time_ns": elapsed_ns,
        "stdout": {"path": stdout_path.name, "sha256": _sha256_file(stdout_path)},
        "stderr": {"path": stderr_path.name, "sha256": _sha256_file(stderr_path)},
        "raw_output": {
            "path": raw_output.name,
            "exists": raw_output.is_file(),
            "sha256": _sha256_file(raw_output) if raw_output.is_file() else None,
        },
        "observation": normalized_observation,
        "parse_error": parse_error,
        "fair_runtime_protocol": frozen,
        "fair_runtime": runtime_receipt,
        "runtime_probe": {
            "path": probe_path.name,
            "sha256": _sha256_file(probe_path) if probe_path.is_file() else None,
            "observation": child_probe,
        },
        "status": "PROCESS_VALID" if process_valid else "INCONCLUSIVE",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "external_claim_authorized": False,
        "claim_blockers": [
            *(["runtime_timeout"] if timed_out else []),
            *(["scheduled_run_inconclusive"] if not process_valid else []),
            "independent_execution_witness_missing",
            "independent_gold_receipt_missing",
            "frozen_public_case_ledger_missing",
        ],
    }
    receipt_path = root / "kaya-fair-runtime-receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def verify_kaya_scheduled_fair_runtime_run_receipt(
    receipt_path: Path | str,
) -> dict[str, object]:
    """Reopen one Kaya scheduled run without granting score or comparison use."""
    path = Path(receipt_path).resolve()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalLeagueError("Kaya fair-runtime receipt is unreadable") from exc
    fields = {
        "schema", "comparator", "input", "conversion_receipt",
        "participant_conformance", "natural_case_parity", "command", "exit_code",
        "wall_time_ns", "stdout", "stderr", "raw_output", "observation",
        "parse_error", "fair_runtime_protocol", "fair_runtime", "runtime_probe",
        "status", "claim_status", "scoreable", "comparison_eligible",
        "external_claim_authorized", "claim_blockers",
    }
    if (
        not isinstance(stored, Mapping)
        or set(stored) != fields
        or stored.get("schema") != KAYA_FAIR_RUNTIME_RUN_SCHEMA
        or stored.get("status") != "PROCESS_VALID"
        or stored.get("claim_status") != "NOT_SCOREABLE"
        or stored.get("scoreable") is not False
        or stored.get("comparison_eligible") is not False
        or stored.get("external_claim_authorized") is not False
        or stored.get("parse_error") is not None
        or stored.get("exit_code") != 0
    ):
        raise ExternalLeagueError("Kaya fair-runtime identity, status, or claim boundary is invalid")
    frozen = _validate_frozen_runtime_protocol(stored.get("fair_runtime_protocol"))
    comparator = stored.get("comparator")
    input_ref = stored.get("input")
    conversion_ref = stored.get("conversion_receipt")
    participant_ref = stored.get("participant_conformance")
    parity_ref = stored.get("natural_case_parity")
    if not all(isinstance(value, Mapping) for value in (
        comparator, input_ref, conversion_ref, participant_ref, parity_ref,
    )):
        raise ExternalLeagueError("Kaya fair-runtime provenance bindings are invalid")
    if (
        set(comparator) != {
            "name", "repository_url", "revision", "tree", "license", "binary",
            "binary_sha256", "working_directory", "source_checkout", "upstream",
        }
        or comparator.get("name") != KAYA_DIRECT_PARTICIPANT_ID
        or comparator.get("repository_url") != KAYA_REPOSITORY_URL
        or comparator.get("revision") != KAYA_SOURCE_REVISION
        or comparator.get("tree") != KAYA_SOURCE_TREE
        or comparator.get("license") != "BSD-3-Clause"
        or comparator.get("working_directory") != str(path.parent)
    ):
        raise ExternalLeagueError("Kaya fair-runtime comparator provenance is invalid")
    python = Path(str(comparator.get("binary", ""))).absolute()
    checkout = Path(str(comparator.get("source_checkout", ""))).resolve()
    if (
        not python.is_file()
        or python.parent.name != "bin"
        or not python.resolve().is_file()
        or comparator.get("binary_sha256") != _sha256_file(python.resolve())
        or comparator.get("binary_sha256") != KAYA_PYTHON_SHA256
        or comparator.get("upstream") != _audit_kaya_source_checkout(checkout)
    ):
        raise ExternalLeagueError("Kaya fair-runtime source or Python provenance drifted")
    if set(input_ref) != {"path", "sha256"} or set(conversion_ref) != {
        "path", "sha256", "renderer_rgb_sha256",
    }:
        raise ExternalLeagueError("Kaya fair-runtime canonical input binding is invalid")
    video = Path(str(input_ref.get("path", ""))).resolve()
    conversion = Path(str(conversion_ref.get("path", ""))).resolve()
    if (
        not video.is_file()
        or input_ref.get("sha256") != _sha256_file(video)
        or not conversion.is_file()
        or conversion_ref.get("sha256") != _sha256_file(conversion)
    ):
        raise ExternalLeagueError("Kaya fair-runtime canonical input hash mismatches")
    conversion_payload = _load_conversion_receipt(conversion, video)
    if conversion_ref.get("renderer_rgb_sha256") != conversion_payload["renderer_rgb"]["raw_sha256"]:
        raise ExternalLeagueError("Kaya fair-runtime conversion binding drifted")
    for label, reference in (("participant", participant_ref), ("parity", parity_ref)):
        if set(reference) != {"path", "sha256"}:
            raise ExternalLeagueError(f"Kaya fair-runtime {label} reference is invalid")
        reference_path = Path(str(reference.get("path", ""))).resolve()
        if not reference_path.is_file() or reference.get("sha256") != _sha256_file(reference_path):
            raise ExternalLeagueError(f"Kaya fair-runtime {label} reference hash mismatches")
    _kaya_fair_runtime_prerequisites(
        str(participant_ref["path"]),
        str(parity_ref["path"]),
        checkout=checkout,
        python=python,
        canonical_video=video,
        conversion_receipt=conversion,
    )
    for label, filename in (("stdout", "stdout.bin"), ("stderr", "stderr.bin")):
        reference = stored.get(label)
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ExternalLeagueError(f"Kaya fair-runtime {label} reference is invalid")
        artifact = _resolve_run_owned_artifact(path, reference["path"], label=f"Kaya {label}")
        if artifact != path.parent / filename or reference.get("sha256") != _sha256_file(artifact):
            raise ExternalLeagueError(f"Kaya fair-runtime {label} hash mismatches")
    raw_ref = stored.get("raw_output")
    if not isinstance(raw_ref, Mapping) or set(raw_ref) != {"path", "exists", "sha256"}:
        raise ExternalLeagueError("Kaya fair-runtime raw output reference is invalid")
    raw = _resolve_run_owned_artifact(path, raw_ref["path"], label="Kaya raw output")
    if (
        raw != path.parent / "child-output.json"
        or raw_ref.get("exists") is not True
        or raw_ref.get("sha256") != _sha256_file(raw)
    ):
        raise ExternalLeagueError("Kaya fair-runtime raw output hash mismatches")
    process_ref = {
        "output": {"path": str(raw), "exists": True, "sha256": _sha256_file(raw)}
    }
    validated = _validate_kaya_child_output(
        _load_kaya_child_output(process_ref),
        mode="native",
        reuse_count=1,
        checkout=checkout,
        environment_root=python.parent.parent.resolve(),
        video=video,
        conversion=conversion,
        direct_manifest=None,
    )
    observation = _normalize_kaya_fair_runtime_output(validated)
    if stored.get("observation") != observation:
        raise ExternalLeagueError("Kaya fair-runtime normalized observation diverges from raw output")
    runtime = stored.get("fair_runtime")
    if not isinstance(runtime, Mapping):
        raise ExternalLeagueError("Kaya fair-runtime run ledger is missing")
    expected_runtime_fields = {
        "schema", "protocol_sha256", "measurement_boundary",
        "environment_policy_sha256", "observed_environment", "timeout_seconds",
        "scheduled_repeat_ordinal", "schedule_binding", "attempt_ordinal",
        "retry_count", "retry_policy", "started_monotonic_ns",
        "finished_monotonic_ns", "wall_time_ns", "timed_out",
        "input_identity_sha256", "normalized_terminal_observation",
    }
    started = runtime.get("started_monotonic_ns")
    finished = runtime.get("finished_monotonic_ns")
    wall_time = runtime.get("wall_time_ns")
    ordinal = runtime.get("scheduled_repeat_ordinal")
    binding = runtime.get("schedule_binding")
    terminal = runtime.get("normalized_terminal_observation")
    if (
        set(runtime) != expected_runtime_fields
        or runtime.get("schema") != FAIR_RUNTIME_RUN_SCHEMA
        or runtime.get("protocol_sha256") != _canonical_json_sha256(frozen)
        or runtime.get("measurement_boundary") != FAIR_RUNTIME_BOUNDARY
        or runtime.get("environment_policy_sha256") != _runtime_environment_sha256(frozen)
        or runtime.get("timeout_seconds") != frozen["budget"]["timeout_seconds"]
        or ordinal not in {1, 2, 3}
        or runtime.get("attempt_ordinal") != 1
        or runtime.get("retry_count") != 0
        or runtime.get("retry_policy") != "NO_RETRY"
        or runtime.get("timed_out") is not False
        or runtime.get("input_identity_sha256") != _sha256_file(video)
        or isinstance(started, bool) or not isinstance(started, int) or started < 0
        or isinstance(finished, bool) or not isinstance(finished, int) or finished <= started
        or isinstance(wall_time, bool) or not isinstance(wall_time, int)
        or wall_time != finished - started
        or stored.get("wall_time_ns") != wall_time
        or wall_time > int(frozen["budget"]["timeout_seconds"]) * 1_000_000_000
        or not isinstance(binding, Mapping)
        or not isinstance(terminal, Mapping)
        or terminal != _normalized_terminal_identity(observation, normalizer="kaya-native-exact-v1")
    ):
        raise ExternalLeagueError("Kaya fair-runtime timing, budget, retry, or normalization ledger is invalid")
    expected_binding = _load_schedule_assignment(
        binding.get("path"),
        schedule_slot=binding.get("slot"),
        protocol=frozen,
        comparator=KAYA_DIRECT_PARTICIPANT_ID,
        repeat_ordinal=int(ordinal),
        input_sha256=_sha256_file(video),
    )
    if expected_binding != dict(binding):
        raise ExternalLeagueError("Kaya fair-runtime schedule binding drifted")
    probe_ref = stored.get("runtime_probe")
    if not isinstance(probe_ref, Mapping) or set(probe_ref) != {"path", "sha256", "observation"}:
        raise ExternalLeagueError("Kaya fair-runtime probe reference is invalid")
    probe_path = _resolve_run_owned_artifact(path, probe_ref["path"], label="Kaya runtime probe")
    probe = _load_child_runtime_probe(probe_path)
    if (
        probe_path != path.parent / "runtime-probe.json"
        or probe_ref.get("sha256") != _sha256_file(probe_path)
        or probe_ref.get("observation") != probe
        or not _observed_environment_matches_protocol(
            runtime.get("observed_environment"), frozen, _sha256_file(video), binding,
        )
        or not _child_probe_and_command_are_bound(
            path, stored, runtime, frozen, KAYA_DIRECT_PARTICIPANT_ID,
        )
    ):
        raise ExternalLeagueError("Kaya fair-runtime probe, environment, or command is unbound")
    child_timing = probe.get("child_timing")
    if (
        not isinstance(child_timing, Mapping)
        or not started <= child_timing.get("probe_started_monotonic_ns", -1)
        <= child_timing.get("tool_started_monotonic_ns", -1)
        < child_timing.get("tool_finished_monotonic_ns", -1) <= finished
    ):
        raise ExternalLeagueError("Kaya fair-runtime child timing escapes the measured boundary")
    return dict(stored)
