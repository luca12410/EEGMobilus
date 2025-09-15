param(
  [string]$topic = "/cmd_vel",
  [int]$domain = 23,
  [int]$port = 9999,
  [switch]$NoGazebo,
  [switch]$EchoTopic  # finestra con ros2 topic echo
)

$ErrorActionPreference = "Stop"

# Paths
$projWin = (Resolve-Path ".").Path
$projWsl = "/mnt/" + $projWin.Substring(0,1).ToLower() + $projWin.Substring(2).Replace("\","/")

# IP WSL
$wslIp = (wsl.exe hostname -I).Split()[0]
if (-not $wslIp) { throw "Impossibile ottenere IP WSL" }

Write-Host "[run] Progetto:     $projWin"
Write-Host "[run] Progetto WSL: $projWsl"
Write-Host "[run] WSL IP:       $wslIp"
Write-Host "[run] ROS_TOPIC:    $topic"
Write-Host "[run] ROS_DOMAIN:   $domain"

function Start-WSLWindow([string]$title, [string]$bashCmd) {
  # Costruisce: cmd.exe /k wsl.exe bash -lc "<bashCmd>"
  $wrapped = "wsl.exe bash -lc `"$bashCmd`""
  Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $wrapped) -WindowStyle Normal
  Write-Host "[run] Avviato: $title"
}

function Start-PSWindow([string]$title, [string]$psCmd) {
  Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit","-Command", $psCmd) -WorkingDirectory $projWin -WindowStyle Normal
  Write-Host "[run] Avviato: $title"
}

# 1) Gazebo (WSL) — GUI se WSLg/X, altrimenti vedrai i log
if (-not $NoGazebo) {
  $gz = @(
    "source /opt/ros/humble/setup.bash",
    "export ROS_DOMAIN_ID=$domain",
    "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
    "export TURTLEBOT3_MODEL=burger",
    "ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py"
  ) -join " && "
  Start-WSLWindow "Gazebo (WSL)" $gz
}

# 2) Bridge UDP→ROS2 (WSL)
$br = @(
  "source /opt/ros/humble/setup.bash",
  "export ROS_DOMAIN_ID=$domain",
  "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
  "export CMDVEL_TOPIC='$topic'",
  "cd '$projWsl'",
  "python3 udp_to_ros2_cmdvel.py"
) -join " && "
Start-WSLWindow "Bridge UDP→ROS2" $br

# 3) (opz) echo del topic (WSL)
if ($EchoTopic) {
  $ec = @(
    "source /opt/ros/humble/setup.bash",
    "export ROS_DOMAIN_ID=$domain",
    "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
    "ros2 topic echo $topic geometry_msgs/msg/Twist"
  ) -join " && "
  Start-WSLWindow "ros2 topic echo $topic" $ec
}

# 4) App (Windows)
$ps = @(
  "Set-Location '$projWin'",
  "`$env:EEG_UDP_TARGET = '${wslIp}:$port'",
  "Write-Host '[app] EEG_UDP_TARGET=' `$env:EEG_UDP_TARGET",
  "if (Test-Path .\.venv-win\Scripts\Activate.ps1) { . .\.venv-win\Scripts\Activate.ps1 }",
  "python3.11 .\main.py"
) -join "; "
Start-PSWindow "EEG App (Windows)" $ps
