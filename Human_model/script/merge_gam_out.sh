#!/bin/bash

# 参数检查
if [ $# -ne 2 ]; then
    echo "Usage: $0 NAME EXT"
    exit 1
fi

NAME=$1
EXT=$2
TOTAL=$(ls $NAME.*.$EXT | wc -l)

# 创建初始文件
cut -f1 -d, $NAME.1.$EXT > $NAME.$EXT
cp -sn $NAME.1.$EXT $NAME.1.$EXT.tmp

# 创建临时文件
seq 2 $TOTAL | xargs -n 1 -I %1 bash -c "cut -f2- -d, $NAME.%1.$EXT > $NAME.%1.$EXT.tmp"

# 带重试的paste命令
max_retries=4
retry_delay=15
attempt=0
success=0

while [ $attempt -lt $max_retries ] && [ $success -eq 0 ]; do
    # 尝试执行paste命令
    if seq 1 $TOTAL | xargs -n 1 -I %1 echo "$NAME.%1.$EXT.tmp" | xargs paste -d, > $NAME.$EXT 2>/dev/null; then
        success=1
    else
        attempt=$((attempt+1))
        if [ $attempt -lt $max_retries ]; then
            echo "paste命令失败 (尝试 $attempt/$max_retries), 等待 ${retry_delay}秒后重试..."
            sleep $retry_delay
        fi
    fi
done

# 检查是否成功
if [ $success -eq 0 ]; then
    echo "错误: paste命令在 $max_retries 次尝试后仍然失败"
    # 清理临时文件
    seq 1 $TOTAL | xargs -n 1 -I %1 rm -f $NAME.%1.$EXT.tmp
    exit 1
fi

# 清理临时文件
seq 1 $TOTAL | xargs -n 1 -I %1 rm $NAME.%1.$EXT.tmp
echo "文件合并成功: $NAME.$EXT"
exit 0