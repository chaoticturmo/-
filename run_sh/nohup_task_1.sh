#!/bin/bash 

# ================= 配置区 =================
# 获取当前时间，用于生成唯一的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_log/task_1_log_${TIMESTAMP}.log"

echo "🚀 开始后台训练任务..."
echo "📄 日志文件: $LOG_FILE"

# 使用 nohup 运行，并将标准输出(stdout)和错误输出(stderr)都重定向到日志文件
nohup python Task_1_Fairness_Analysis.py \
  > "$LOG_FILE" 2>&1 &

# 打印进程 ID，方便你后续杀进程
PID=$!
echo "✅ 任务已启动，PID: $PID"
echo "👉 你可以使用以下命令查看实时日志："
echo "   tail -f $LOG_FILE"