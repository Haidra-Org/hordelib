#!/bin/bash

# Check if all required arguments were provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <number_of_instances> <version>"
    echo "Example: $0 20 v26"
    exit 1
fi

# Arguments
N=$1
VERSION=$2

# Validate number of instances is a positive number
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "Error: First argument must be a positive number"
    exit 1
fi

# Validate version starts with 'v'
if ! [[ "$VERSION" =~ ^v[0-9]+$ ]]; then
    echo "Error: Version must be in format 'v<number>' (e.g., v26)"
    exit 1
fi

# Array to store PIDs of all children
declare -a PIDS

# Function to clean up processes on exit
cleanup() {
    echo -e "\nReceived interrupt signal. Stopping all training processes..."
    # Kill all children in the process group
    for pid in "${PIDS[@]}"; do
        pkill -2 -P "$pid" 2>/dev/null
        kill -2 "$pid" 2>/dev/null
        echo "kill sent to ${pid}"
    done
    exit 1
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# Get current datetime for log files
DATETIME=$(date "+%Y%m%d_%H%M")

# Counter for naming log files
counter=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting $N training instances with version $VERSION..."

# Start N instances in parallel
for i in $(seq 1 $N); do
    LOG_FILE="logs/train_${VERSION}_${DATETIME}_${counter}.log"
    # Run each instance and pipe output to log file
    python train.py -ev $VERSION > "$LOG_FILE" 2>&1 &
    # Store the PID of the python process
    PIDS+=($!)
    echo "Started instance $counter with version $VERSION (log: $LOG_FILE, PID: ${PIDS[-1]})"
    ((counter++))
done

echo "All processes started. Press Ctrl+C to stop all instances."

# Wait for all background processes to complete
wait

echo "All training instances have completed"

# Check exit status of all processes
for pid in "${PIDS[@]}"; do
    wait $pid || echo "Process $pid failed"
done

echo "Log files are stored in the logs directory with format: train_<version>_<datetime>_<instance>.log"
