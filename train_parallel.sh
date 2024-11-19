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

# Counter for naming log files
counter=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting $N training instances with version $VERSION..."

# Start N instances in parallel
for i in $(seq 1 $N); do
    # Run each instance with its output redirected to a log file
    python train.py -ev $VERSION > "logs/train_${VERSION}_${counter}.log" 2>&1 &
    echo "Started instance $counter with version $VERSION"
    ((counter++))
done

# Wait for all background processes to complete
wait

echo "All training instances have completed"

# Check exit status of all processes
for job in $(jobs -p); do
    wait $job || echo "Process $job failed"
done
