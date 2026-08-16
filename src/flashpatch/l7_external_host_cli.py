import argparse
import sys
from pathlib import Path

from flashpatch.l7_executor import CandidateStartGateInputs, run_external_host_9_slot_executor
from flashpatch.l7_external_host import ExternalHostWitnessError

def main() -> int:
    parser = argparse.ArgumentParser(description="Execute L7 external host 9-slot request")
    parser.add_argument("--request", type=Path, required=True, help="Path to frozen request v2 JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--intake-receipt", type=Path, required=True, help="Native-main blind intake receipt")
    parser.add_argument("--return-root", type=Path, required=True, help="Native-main independent gold return root")
    parser.add_argument("--packet-manifest", type=Path, required=True, help="Native-main adjudicator packet manifest")
    parser.add_argument("--start-receipt", type=Path, required=True, help="Native-main candidate-start gate receipt")
    args = parser.parse_args()

    try:
        receipt_path = run_external_host_9_slot_executor(
            args.request,
            args.output_dir,
            candidate_start_gate=CandidateStartGateInputs(
                intake_receipt=args.intake_receipt,
                return_root=args.return_root,
                packet_manifest=args.packet_manifest,
                start_receipt=args.start_receipt,
            ),
        )
        print(f"Success: {receipt_path}")
        return 0
    except ExternalHostWitnessError as exc:
        print(f"FAIL_CLOSED:executor:{exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
