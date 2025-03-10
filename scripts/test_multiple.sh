parent_folder="./wandb/wandb_cluster/"

# Using 'find' to get subfolders and store them in an array
subfolders=($(find "$parent_folder" -mindepth 1 -maxdepth 1 -type d -exec realpath {} \;))

# Iterate over the array and print each subfolder
for subfolder in "${subfolders[@]:0:7}"; do
  #echo "$subfolder"
  python /home/i53/student/pfeil/aloha_platform_simulation/ai/movement-primitive-diffusion/scripts/test_multiple_agents_checkpoints_in_env.py  --config-name test_multiple_agents_checkpoints_in_env.yaml +run_folder="$subfolder"
done

