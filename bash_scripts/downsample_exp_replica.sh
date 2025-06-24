#!/bin/bash
# filepath: /home/xiongyuxiang/SplaTAM/bash_scripts/downsampling.sh

# 创建实验结果目录
RESULTS_DIR="./experiment_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

# 初始化摘要文件，添加新增指标
echo "config,ate_rmse,psnr,depth_rmse,depth_l1,ms_ssim,lpips" > $SUMMARY_FILE

# 配置列表
CONFIGS=(
    "baseline"
    "tile8_harris" 
    "tile8_random" 
    "tile8_uniform"
    "tile32_harris"
    "tile32_random"
    "tile32_uniform"
)

# 运行单个实验
run_experiment() {
    local config=$1
    
    # 解析配置
    if [ "$config" == "baseline" ]; then
        sampling_args=""
    else
        tile_size=$(echo $config | cut -d'_' -f1 | sed 's/tile//')
        sparse_fn=$(echo $config | cut -d'_' -f2)
        sampling_args="--use_sampling --tile_size $tile_size --sparse_fn $sparse_fn"
    fi
    
    echo "[$config] start"
    
    # 运行实验
    python scripts/splatam.py configs/replica/splatam.py $sampling_args > "$RESULTS_DIR/${config}.log" 2>&1
    
    ate_rmse=$(grep "Final Average ATE RMSE" "$RESULTS_DIR/${config}.log" | awk '{print $5}')
    psnr=$(grep "Average PSNR" "$RESULTS_DIR/${config}.log" | awk '{print $3}')
    depth_rmse=$(grep "Average Depth RMSE" "$RESULTS_DIR/${config}.log" | awk '{print $4}')
    depth_l1=$(grep "Average Depth L1" "$RESULTS_DIR/${config}.log" | awk '{print $4}')
    ms_ssim=$(grep "Average MS-SSIM" "$RESULTS_DIR/${config}.log" | awk '{print $3}')
    lpips=$(grep "Average LPIPS" "$RESULTS_DIR/${config}.log" | awk '{print $3}')
    
    # 保存到摘要文件
    echo "$config,$ate_rmse,$psnr,$depth_rmse,$depth_l1,$ms_ssim,$lpips" >> $SUMMARY_FILE
    
    echo "[$config] finish"
}

# 运行所有配置
for config in "${CONFIGS[@]}"; do
    run_experiment "$config"
done

# 生成简单的结果摘要
cat $SUMMARY_FILE