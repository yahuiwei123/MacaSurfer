import pandas as pd
import numpy as np
import os
import sys
import argparse
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel

meta_cols = {"subject_id", "age", "sex", "site", "breed", "weight (kg)", "atlas", "hemisphere"}


def get_model_batch_effects_info(model):
    """
    Extract batch effects information from the model.
    Returns a dict mapping batch effect name to list of known values.
    """
    known_batch_effects = {}
    try:
        # Method 1: Check if model has batch_effects attribute directly
        if hasattr(model, 'batch_effects') and model.batch_effects is not None:
            be = model.batch_effects
            if hasattr(be, 'unique'):
                for col in be.columns if hasattr(be, 'columns') else []:
                    known_batch_effects[col] = list(be[col].unique())
            elif isinstance(be, dict):
                for be_name, be_values in be.items():
                    if hasattr(be_values, 'unique'):
                        known_batch_effects[be_name] = list(be_values.unique())
                    elif isinstance(be_values, (list, np.ndarray)):
                        known_batch_effects[be_name] = list(np.unique(be_values))

        # Method 2: Check if model has _batch_effects (private attribute)
        if not known_batch_effects and hasattr(model, '_batch_effects') and model._batch_effects is not None:
            be = model._batch_effects
            if isinstance(be, dict):
                for be_name, be_values in be.items():
                    if hasattr(be_values, 'unique'):
                        known_batch_effects[be_name] = list(be_values.unique())
                    elif isinstance(be_values, (list, np.ndarray)):
                        known_batch_effects[be_name] = list(np.unique(be_values))

        # Method 3: Check model's saved data info
        if not known_batch_effects and hasattr(model, 'saved_data_info'):
            sdi = model.saved_data_info
            if hasattr(sdi, 'batch_effects') and sdi.batch_effects is not None:
                for be_name, be_values in sdi.batch_effects.items():
                    if hasattr(be_values, 'unique'):
                        known_batch_effects[be_name] = list(be_values.unique())
                    elif isinstance(be_values, (list, np.ndarray)):
                        known_batch_effects[be_name] = list(np.unique(be_values))

        # Method 4: Check if model has a scaler or encoder with batch effects info
        if not known_batch_effects and hasattr(model, 'batch_effects_encoder'):
            encoder = model.batch_effects_encoder
            if hasattr(encoder, 'categories_'):
                # sklearn-style encoder
                for i, col in enumerate(getattr(encoder, 'feature_names_in_', ['sex', 'site'])):
                    known_batch_effects[col] = list(encoder.categories_[i])
            elif hasattr(encoder, 'classes_'):
                known_batch_effects['batch_effect'] = list(encoder.classes_)

        # Method 5: Try to inspect model's configuration
        if not known_batch_effects and hasattr(model, 'config'):
            config = model.config
            if hasattr(config, 'batch_effects'):
                for be_name in config.batch_effects:
                    known_batch_effects[be_name] = []

        # Method 6: Look for any attribute containing 'batch' or 'site' or 'sex'
        if not known_batch_effects:
            for attr_name in dir(model):
                if 'batch' in attr_name.lower() or 'site' in attr_name.lower():
                    attr = getattr(model, attr_name, None)
                    if attr is not None and not callable(attr):
                        print(f"  Found potential batch effect attribute: {attr_name}")
                        if hasattr(attr, 'unique'):
                            known_batch_effects[attr_name] = list(attr.unique())
                        elif isinstance(attr, (list, np.ndarray)):
                            known_batch_effects[attr_name] = list(np.unique(attr))

    except Exception as e:
        print(f"Warning: Could not extract batch effects info from model: {e}")
        import traceback
        traceback.print_exc()

    return known_batch_effects


def harmonize_batch_effects(data, batch_effects, known_batch_effects):
    """
    Harmonize batch effects in the data to match known values from the model.
    Unknown values will be replaced with the first known value.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    batch_effects : list
        List of batch effect column names
    known_batch_effects : dict
        Dict mapping batch effect name to list of known values

    Returns:
    --------
    data : pd.DataFrame
        Data with harmonized batch effects
    """
    data = data.copy()

    for be_name in batch_effects:
        if be_name not in data.columns:
            continue

        if be_name in known_batch_effects and known_batch_effects[be_name]:
            known_values = known_batch_effects[be_name]
            # Find values in data that are not in known values
            unique_data_values = data[be_name].unique()
            unknown_values = [v for v in unique_data_values if v not in known_values]

            if unknown_values:
                # Use the first known value as default
                default_value = known_values[0]
                print(f"Warning: Unknown batch effect values for '{be_name}': {unknown_values}")
                print(f"  Replacing with default: '{default_value}'")
                data.loc[data[be_name].isin(unknown_values), be_name] = default_value
        else:
            # If we don't have known values for this batch effect, try to use a common default
            print(f"Warning: No known values for batch effect '{be_name}', using first unique value as default")
            unique_vals = data[be_name].unique()
            if len(unique_vals) > 0:
                default_value = unique_vals[0]
                data[be_name] = default_value

    return data


def load_model_and_predict(atlas, hemi, metric, new_data_path, save_base_dir="resources/blr/save_dir"):
    """
    加载已保存的模型并对新数据进行预测

    Parameters:
    -----------
    atlas : str
        图谱名称
    hemi : str
        半球 ('L' or 'R')
    metric : str
        指标 ('cortvol', 'curvature', 'subvol', 'sulc', 'thickness')
    new_data_path : str
        新数据 CSV 文件路径
    save_base_dir : str
        模型保存的基础目录

    Returns:
    --------
    model : NormativeModel
        加载的模型
    results : NormData
        包含预测结果的对象
    results_df : pd.DataFrame
        预测结果的 DataFrame
    """

    # 1. 加载已保存的模型
    model_path = f"{save_base_dir}/{atlas}/{hemi}/{metric}"
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = NormativeModel.load(model_path)
    print(f"Model loaded successfully!")
    print(f"  Response vars: {model.response_vars}")
    print(f"  Is fitted: {model.is_fitted}")

    # Get known batch effects from model
    known_batch_effects = get_model_batch_effects_info(model)
    print(f"  Known batch effects: {known_batch_effects}")

    # 2. 读取新数据
    new_data = pd.read_csv(new_data_path)
    print(f"Loaded data from: {new_data_path}")
    print(f"  Shape: {new_data.shape}")
    print(f"  Columns: {new_data.columns.tolist()}")

    # 3. 获取与训练时相同的 response_vars（从模型中获取）
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = model.response_vars  # 使用模型中保存的 response_vars

    # 4. 过滤数据（确保 response_vars 中的列存在且为数值类型）
    available_response_vars = [
        col for col in response_vars
        if col in new_data.columns and np.issubdtype(new_data[col].dtype, np.number)
    ]

    # 检查是否有缺失的 response_vars
    missing_vars = set(response_vars) - set(available_response_vars)
    if missing_vars:
        print(f"Warning: Missing response vars in new data: {missing_vars}")

    # 5. 过滤 NaN 和 inf
    def filter_data(data, covariates, batch_effects, response_vars):
        cols_to_check = covariates + batch_effects + response_vars
        cols_to_check = [c for c in cols_to_check if c in data.columns]
        data_filtered = data.dropna(subset=cols_to_check)
        data_filtered = data_filtered[~data_filtered[cols_to_check].isin([np.inf, -np.inf]).any(axis=1)]
        return data_filtered

    data_filtered = filter_data(new_data, covariates, batch_effects, available_response_vars)
    print(f"Data filtered: {len(new_data)} -> {len(data_filtered)} rows")

    if len(data_filtered) == 0:
        raise ValueError("No valid data rows after filtering. Please check input data.")

    # 6. Harmonize batch effects to match model's known values
    data_filtered = harmonize_batch_effects(data_filtered, batch_effects, known_batch_effects)

    # 7. 创建 NormData 对象
    norm_data = NormData.from_dataframe(
        name="predict",
        dataframe=data_filtered,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=available_response_vars
    )

    print("=====================")
    print(norm_data)

    # 8. 进行预测
    print("Predicting...")
    results = model.predict(norm_data)

    # 9. 提取结果
    # results 包含: Zscores, centiles, logp, yhat 等
    results_df = results.to_dataframe()

    print(f"Prediction completed!")
    print(f"  Columns in results: {results_df.columns.tolist()}")

    return model, results, results_df


def single_predict(atlas, hemi, metric, input_csv, norm_model_dir, out_dir):
    """
    单次预测：针对单个 atlas/hemi/metric 组合

    Parameters:
    -----------
    atlas : str
        图谱名称
    hemi : str
        半球 ('L' or 'R')
    metric : str
        指标 ('cortvol', 'curvature', 'subvol', 'sulc', 'thickness')
    input_csv : str
        输入 CSV 文件路径（由 roi_stats.py 生成）
    norm_model_dir : str
        Normative model 目录
    out_dir : str
        输出目录
    """
    print(f"\n{'='*60}")
    print(f"Starting prediction: {atlas}/{hemi}/{metric}")
    print('='*60)

    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found: {input_csv}")
        return None

    try:
        _, _, results_df = load_model_and_predict(
            atlas=atlas,
            hemi=hemi,
            metric=metric,
            new_data_path=input_csv,
            save_base_dir=norm_model_dir
        )

        # 保存结果
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{metric}_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return results_df

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Normative model prediction for ROI statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python norm_predict.py --atlas MBNA124 --hemi L --metric thickness \\
      --input_csv stats/sub01__thickness__MBNA124__L.csv \\
      --norm_model_dir resources/blr/save_dir \\
      --out_dir predictions
        """
    )

    parser.add_argument("--atlas", type=str, required=True,
                        help="Atlas name (e.g., MBNA124, Modalities, M129, M132)")
    parser.add_argument("--hemi", type=str, required=True, choices=['L', 'R'],
                        help="Hemisphere (L or R)")
    parser.add_argument("--metric", type=str, required=True,
                        choices=['cortvol', 'curvature', 'subvol', 'sulc', 'thickness'],
                        help="Metric type")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input CSV file path (generated by roi_stats.py)")
    parser.add_argument("--norm_model_dir", type=str, required=True,
                        help="Normative model directory path")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for predictions")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}")
        sys.exit(1)

    result = single_predict(
        atlas=args.atlas,
        hemi=args.hemi,
        metric=args.metric,
        input_csv=args.input_csv,
        norm_model_dir=args.norm_model_dir,
        out_dir=args.out_dir
    )

    if result is not None:
        print("\nPrediction completed successfully!")
    else:
        print("\nPrediction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
