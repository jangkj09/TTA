#!/bin/bash

unset -v device
while getopts "d:" opt; do
  case $opt in
    d)
      device=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
if [ -z "$device" ]; then
  echo "Usage: $0 -d <device_id>"
  exit 1
fi
# Ensure the device is a valid integer
if ! [[ "$device" =~ ^[0-9]+$ ]]; then
  echo "Error: Device ID must be a valid integer." >&2
  exit 1
fi  
# Set the CUDA device
echo "Setting CUDA_VISIBLE_DEVICES to $device"
# This will allow the script to run on the specified GPU device
# If you want to run on CPU, you can set device to -1 and modify the script accordingly    
export CUDA_VISIBLE_DEVICES=$device

python3 audio/whisper_librispeech_bt.py


