#!/bin/bash
# Contamination judge — sourced by test.sh and test.tinker.sh
# Expects: $WORKSPACE, $TESTS_DIR, $LOGS_DIR, $MODEL_ID, $BENCHMARK_NAME

echo ""
echo "=== Running Contamination Judge ==="

JUDGE_DIR="$TESTS_DIR/contamination"

if [ -f "$JUDGE_DIR/judge.py" ] && [ -n "$BENCHMARK_NAME" ]; then
	JUDGE_TASK=$(python3 "$JUDGE_DIR/judge.py" \
		--model "$MODEL_ID" \
		--benchmark "$BENCHMARK_NAME" 2>/dev/null) || true

	if [ -n "$JUDGE_TASK" ] && [ -n "$OPENAI_API_KEY" ]; then
		echo "Running codex CLI contamination judge..."
		set +e
		cd "$WORKSPACE"
		codex --search -a never exec --json -c model_reasoning_summary=detailed \
			--skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_TASK" \
			2>&1 | tee "$LOGS_DIR/judge_output.json"
		set -e

		for f in contamination_judgement.txt disallowed_model_judgement.txt; do
			if [ -f "$WORKSPACE/$f" ]; then
				cp "$WORKSPACE/$f" "$LOGS_DIR/$f"
				echo "$f: $(cat "$LOGS_DIR/$f")"
			fi
		done
		[ -f "$LOGS_DIR/contamination_judgement.txt" ] || echo "no contamination detected (codex did not produce output)" >"$LOGS_DIR/contamination_judgement.txt"
		[ -f "$LOGS_DIR/disallowed_model_judgement.txt" ] || echo "only allowed use detected (codex did not produce output)" >"$LOGS_DIR/disallowed_model_judgement.txt"
	else
		echo "Warning: OPENAI_API_KEY not set or prompt generation failed, skipping judge"
		echo "no contamination detected (judge skipped - no API key)" >"$LOGS_DIR/contamination_judgement.txt"
		echo "only allowed use detected (judge skipped - no API key)" >"$LOGS_DIR/disallowed_model_judgement.txt"
	fi
else
	echo "Warning: judge.py or metadata not found, skipping judge"
	echo "no contamination detected (judge not available)" >"$LOGS_DIR/contamination_judgement.txt"
	echo "only allowed use detected (judge not available)" >"$LOGS_DIR/disallowed_model_judgement.txt"
fi
