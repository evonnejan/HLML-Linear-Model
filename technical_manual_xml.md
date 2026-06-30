# XML Technical Manual

```xml
<technical_manual>
  <project_overview>
    <purpose>
      This repository implements time-series forecasting for water-level prediction using linear-family models (Linear, NLinear, DLinear, DLinearMix, DLinearMix2), with run-level and segment-level evaluation workflows.
    </purpose>
    <current_goal>
      Train many input-to-target settings, select best checkpoints by test MSE, and analyze horizon/segment behavior using overview and full-inference scripts.
    </current_goal>
    <business_objective>
      Unknown from current code inspection.
    </business_objective>
  </project_overview>

  <current_status>
    <implemented>
      <item>Training and testing pipeline via run.py and exp/exp_Main2.py.</item>
      <item>Segment-aware dataset split/window generation in data_provider/Data_Loader.py.</item>
      <item>Model variants in models/: Linear, NLinear, DLinear, DLinearMix, DLinearMix2.</item>
      <item>Run output artifacts (metrics.npy, horizon/segment CSV files, checkpoints).</item>
      <item>Best-model overview analysis via analyze_best_models_overview.py.</item>
      <item>Full split coverage analysis via analyze_full_inference.py.</item>
    </implemented>
    <available_workflows>
      <item>Single run training+testing.</item>
      <item>Best-run overlay and fanplot generation.</item>
      <item>Full-inference segment-horizon aggregation and top30 overlap diagnostics.</item>
      <item>Run-level visualization utility via visualize.py.</item>
    </available_workflows>
    <workspace_evidence>
      <item>Many run folders under runs/ including DLinear, DLinearMix, DLinearMix2 settings.</item>
      <item>Summary CSV exists: runs/summary/DLinearMix2_summary.csv.</item>
      <item>Analysis outputs exist under analysis/ and analysis/full_inference/.</item>
    </workspace_evidence>
  </current_status>

  <folder_structure>
    <path>run.py</path>
    <path>analyze_best_models_overview.py</path>
    <path>analyze_full_inference.py</path>
    <path>analyze_full_inference_lag.py</path>
    <path>visualize.py</path>
    <path>data_provider/</path>
    <path>exp/</path>
    <path>models/</path>
    <path>utils/</path>
    <path>dataset/</path>
    <path>runs/</path>
    <path>analysis/</path>
    <path>checkpoints/</path>
    <path>test_results/</path>
    <path>Source_Code/</path>
    <missing_manifest>
      <item>README*: not found.</item>
      <item>requirements*.txt: not found.</item>
      <item>pyproject.toml: not found.</item>
    </missing_manifest>
  </folder_structure>

  <tasks>
    <task id="training_testing">
      <description>Run model training and testing with saved run configuration and outputs.</description>
      <entrypoint>run.py</entrypoint>
      <engine>exp/exp_Main2.py::Exp_Main</engine>
      <outputs>
        <item>runs/&lt;setting&gt;/run_args.json</item>
        <item>runs/&lt;setting&gt;/run_cmd.txt</item>
        <item>runs/&lt;setting&gt;/checkpoints/checkpoint.pth</item>
        <item>runs/&lt;setting&gt;/outputs/metrics.npy</item>
        <item>runs/&lt;setting&gt;/outputs/mse_horizon.csv</item>
        <item>runs/&lt;setting&gt;/outputs/mse_segment_combined.csv</item>
        <item>runs/&lt;setting&gt;/outputs/segment_horizon_points.csv.gz</item>
        <item>runs/&lt;setting&gt;/outputs/segment_horizon_rank.csv</item>
        <item>runs/&lt;setting&gt;/outputs/meeting.csv</item>
      </outputs>
    </task>

    <task id="best_models_overview">
      <description>Select best run per input-target pair and generate horizon/segment overview artifacts.</description>
      <entrypoint>analyze_best_models_overview.py</entrypoint>
      <outputs>
        <item>analysis/run_overview.csv</item>
        <item>analysis/horizon_mse_overlay.csv</item>
        <item>analysis/horizon_mse_overlay_relative.csv</item>
        <item>analysis/segment_fanplots/segment_fanplot_draw_records.csv</item>
        <item>analysis/segment_fanplots/segment_fanplot_counts.csv</item>
      </outputs>
    </task>

    <task id="full_inference">
      <description>Select best run per pair and compute full segment-horizon metrics across train/val/test points.</description>
      <entrypoint>analyze_full_inference.py</entrypoint>
      <outputs>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/full_segment_horizon_metrics.csv</item>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/best_horizon_segment_metrics.csv</item>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/lag_horizon_distribution.csv</item>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/top30_mse_full_inference.csv</item>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/top30_corr_full_inference.csv</item>
        <item>analysis/full_inference/&lt;timestamp&gt;__*/top30_overlap_segments_full_inference.csv</item>
        <item>runs/&lt;setting&gt;/outputs/segment_horizon_points_full.csv.gz</item>
      </outputs>
    </task>
  </tasks>

  <core_modules>
    <module path="data_provider/">
      <file path="data_provider/Data_Factory.py">
        <does>Builds dataset and DataLoader for train/val/test/pred modes.</does>
        <functions>
          <function>data_provider(args, flag)</function>
        </functions>
        <inputs>
          <item>args fields: data, root_path, data_path, seq_len, label_len, pred_len, features, target, input_col, exog_col, segment_col, stride_train, stride_eval, embed, freq, train_only, model.</item>
          <item>flag in {train, val, test, pred}.</item>
        </inputs>
        <outputs>
          <item>Dataset instance (Dataset_Custom or Dataset_Pred).</item>
          <item>torch DataLoader.</item>
        </outputs>
      </file>
      <file path="data_provider/Data_Loader.py">
        <does>Implements dataset loading, split, scaling, windowing, and inverse transform.</does>
        <classes>
          <class>Dataset_Custom</class>
          <class>Dataset_Pred</class>
        </classes>
        <key_methods>
          <method>Dataset_Custom.__read_data__</method>
          <method>Dataset_Custom.__getitem__</method>
          <method>Dataset_Custom.__len__</method>
          <method>Dataset_Custom.inverse_transform</method>
        </key_methods>
        <split_logic>
          <item>If segment_col set: segment-based split (train/val/test), with no cross-segment windows.</item>
          <item>Else: row-based fallback split.</item>
        </split_logic>
        <connections>
          <item>Called by experiment classes in exp/.</item>
          <item>Provides segment metadata consumed by test and analysis scripts.</item>
        </connections>
      </file>
    </module>

    <module path="exp/">
      <file path="exp/exp_Basic.py">
        <does>Defines base experiment class and device selection.</does>
        <class>Exp_Basic</class>
        <key_methods>
          <method>_acquire_device</method>
        </key_methods>
      </file>
      <file path="exp/exp_Main2.py">
        <does>Main training/testing implementation currently used by run.py.</does>
        <class>Exp_Main</class>
        <key_methods>
          <method>train</method>
          <method>test</method>
          <method>predict</method>
          <method>_save_horizon_mse</method>
          <method>_save_segment_metrics</method>
        </key_methods>
        <connections>
          <item>Consumes data_provider outputs.</item>
          <item>Builds models from models/.</item>
          <item>Writes outputs consumed by analysis scripts.</item>
        </connections>
      </file>
      <file path="exp/exp_Main.py">
        <does>Alternative/older experiment implementation with similar interface.</does>
        <current_usage>
          Imported by analyze_full_inference.py.
        </current_usage>
      </file>
    </module>

    <module path="models/">
      <file path="models/Linear.py">
        <class>Model</class>
        <io>
          <input>[B, L, C]</input>
          <output>[B, P, C]</output>
        </io>
      </file>
      <file path="models/NLinear.py">
        <class>Model</class>
        <behavior>Subtract last value, forecast, add back.</behavior>
      </file>
      <file path="models/DLinear.py">
        <classes>
          <class>moving_avg</class>
          <class>series_decomp</class>
          <class>Model</class>
        </classes>
      </file>
      <file path="models/DLinearMix.py">
        <class>Model</class>
        <behavior>Early channel mixing then DLinear decomposition.</behavior>
      </file>
      <file path="models/DLinearMix2.py">
        <classes>
          <class>DLinearBranch</class>
          <class>ExogenousEncoder</class>
          <class>HorizonWiseFusion</class>
          <class>FlattenFusion</class>
          <class>Model</class>
        </classes>
        <behavior>Branch-wise DLinear plus exogenous-context fusion.</behavior>
      </file>
    </module>

    <module path="utils/">
      <file path="utils/metrics.py">
        <functions>
          <function>metric</function>
          <function>MSE</function>
          <function>MAE</function>
          <function>RMSE</function>
          <function>MAPE</function>
          <function>MSPE</function>
          <function>RSE</function>
          <function>CORR</function>
        </functions>
      </file>
      <file path="utils/tools.py">
        <classes>
          <class>EarlyStopping</class>
          <class>dotdict</class>
          <class>StandardScaler</class>
        </classes>
        <functions>
          <function>adjust_learning_rate</function>
          <function>visual</function>
          <function>test_params_flop</function>
        </functions>
      </file>
      <file path="utils/timefeatures.py">
        <functions>
          <function>time_features_from_frequency_str</function>
          <function>time_features</function>
        </functions>
      </file>
    </module>

    <script path="run.py">
      <does>CLI entrypoint for training and testing.</does>
      <key_functions>
        <function>_configure_mix_model_args</function>
        <function>main</function>
      </key_functions>
    </script>

    <script path="analyze_full_inference.py">
      <does>Full split inference analysis for best run per pair.</does>
      <key_functions>
        <function>discover_runs</function>
        <function>load_run_info</function>
        <function>run_full_inference</function>
        <function>main</function>
      </key_functions>
    </script>

    <script path="analyze_best_models_overview.py">
      <does>Best-run summary analysis and fanplot generation.</does>
      <key_functions>
        <function>discover_runs</function>
        <function>load_run_info</function>
        <function>plot_horizon_overlay</function>
        <function>plot_segment_fan</function>
        <function>main</function>
      </key_functions>
    </script>
  </core_modules>

  <data_pipeline>
    <expected_data_format>
      <example_file path="dataset/water_level_all.csv">
        <header>date,HL01,HL02,HL03,HL04,HL05,HL06</header>
      </example_file>
      <example_file path="dataset/water_level_rain_all4.csv">
        <header>date,SegmentStart,SegmentEnd,segment_id,WinStart,WinEnd,isRain,HL01,HL02,HL03,HL04,HL05,HL06,StationId,Past10Min,Past1Hr,Past3Hr,Past6Hr,Past12Hr,Past24Hr,Past2Day,Past3Day,Now</header>
      </example_file>
      <required_columns>
        <item>date</item>
        <item>target column from --target</item>
        <item>input column(s) from --input_col (or fallback target in S mode)</item>
      </required_columns>
    </expected_data_format>
    <unknowns>
      <item>Official production dataset and schema contract are unknown from current code inspection.</item>
    </unknowns>
  </data_pipeline>

  <usage>
    <training_testing>
      <command>python run.py --model DLinear --data custom --root_path ./dataset --data_path water_level_rain_all4.csv --features S --input_col HL02 --target HL01 --segment_col segment_id --seq_len 60 --pred_len 15</command>
      <command>python run.py --model DLinearMix2 --data custom --root_path ./dataset --data_path water_level_rain_all4.csv --features S --input_col HL02,HL03 --exog_col isRain --target HL01 --segment_col segment_id --seq_len 60 --pred_len 15</command>
    </training_testing>
    <analysis>
      <command>python analyze_best_models_overview.py --runs_root ./runs --out_dir ./analysis --models DLinear --targets HL01</command>
      <command>python analyze_full_inference.py --runs_root ./runs --out_dir ./analysis --models DLinear --targets HL01</command>
    </analysis>
    <visualization>
      <command>python visualize.py --mode topk --output_root ./runs --k 10</command>
      <command>python visualize.py --mode segment --run_dir ./runs/&lt;setting&gt; --segment 220 --horizon 7 --points_source full</command>
    </visualization>
    <important_cli_args>
      <item>--model, --input_col, --exog_col, --target, --segment_col</item>
      <item>--seq_len, --pred_len, --stride_train, --stride_eval</item>
      <item>--dlinear_kernel_size, --flatten_fusion, --branch_in, --exog_in, --mix_in</item>
      <item>--train_epochs, --batch_size, --learning_rate, --patience, --lradj</item>
      <item>--use_gpu, --use_amp, --use_multi_gpu, --devices</item>
    </important_cli_args>
  </usage>

  <outputs_and_results>
    <run_level>
      <path>runs/&lt;setting&gt;/checkpoints/checkpoint.pth</path>
      <path>runs/&lt;setting&gt;/outputs/metrics.npy</path>
      <path>runs/&lt;setting&gt;/outputs/mse_horizon.csv</path>
      <path>runs/&lt;setting&gt;/outputs/mse_segment_combined.csv</path>
      <path>runs/&lt;setting&gt;/outputs/segment_horizon_points.csv.gz</path>
      <path>runs/&lt;setting&gt;/outputs/segment_horizon_rank.csv</path>
      <path>runs/&lt;setting&gt;/outputs/meeting.csv</path>
    </run_level>
    <aggregate_level>
      <path>analysis/run_overview.csv</path>
      <path>analysis/horizon_mse_overlay.csv</path>
      <path>analysis/full_segment_horizon_metrics.csv</path>
      <path>analysis/top30_mse_full_inference.csv</path>
      <path>analysis/top30_corr_full_inference.csv</path>
      <path>analysis/top30_overlap_segments_full_inference.csv</path>
    </aggregate_level>
  </outputs_and_results>

  <known_issues>
    <issue>run.py uses exp/exp_Main2.py, while analyze_full_inference.py imports exp/exp_Main.py. This can create behavior mismatch.</issue>
    <issue>No dependency manifest detected (README/requirements/pyproject not found).</issue>
    <issue>Analysis scripts require specific existing files; runs missing them are skipped.</issue>
    <issue>MAPE/MSPE divide by true values directly and may be unstable for zero targets.</issue>
    <issue>CORR in utils/metrics.py is scaled by 0.01, which is non-standard.</issue>
    <issue>Legacy and active code coexist (Source_Code/, exp_Main.py, exp_Main2.py), increasing maintenance ambiguity.</issue>
  </known_issues>

  <next_steps>
    <step>Unify canonical experiment engine and make all pipelines use it consistently.</step>
    <step>Add reproducible environment files and dependency pinning.</step>
    <step>Create a formal README with data schema, commands, and output contracts.</step>
    <step>Add automated checks/tests for dataset splitting, metrics, and analysis contracts.</step>
    <step>Define archival policy for legacy folders and historical artifacts.</step>
  </next_steps>

  <important_files_for_future_work>
    <path>run.py</path>
    <path>exp/exp_Main2.py</path>
    <path>data_provider/Data_Loader.py</path>
    <path>models/DLinearMix2.py</path>
    <path>analyze_full_inference.py</path>
    <path>analyze_best_models_overview.py</path>
    <path>runs/summary/DLinearMix2_summary.csv</path>
    <path>analysis/run_overview.csv</path>
    <path>analysis/full_segment_horizon_metrics.csv</path>
  </important_files_for_future_work>

  <questions_for_project_owner>
    <question>Should exp/exp_Main2.py fully replace exp/exp_Main.py, including full-inference analysis?</question>
    <question>What exact Python and package versions produced the trusted baseline results?</question>
    <question>Which dataset file is the canonical production input for new experiments?</question>
    <question>Should CORR remain scaled by 0.01 or be changed to standard Pearson scaling?</question>
    <question>Is Source_Code/ historical only, or still part of active development?</question>
  </questions_for_project_owner>
</technical_manual>
```