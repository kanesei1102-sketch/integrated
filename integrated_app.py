import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# ---------------------------------------------------------
# 0. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("""
**統計解析から論文グレードのグラフ作成までを自動化する統合ツール (Pro Ver.)**
データの性質を自動診断し、最適な検定を選択。有意差バー付きのグラフを一瞬で作成します。
""")

# ---------------------------------------------------------
# 1. データ入力セクション (CSV & Manual)
# ---------------------------------------------------------
st.subheader("1. データ入力")
tab_manual, tab_csv = st.tabs(["✍️ 手動入力", "📂 CSVアップロード"])

data_dict = {}

# --- A. 手動入力モード ---
with tab_manual:
    if 'g_count' not in st.session_state: st.session_state.g_count = 3
    
    col_ctrl, _ = st.columns([1, 5])
    with col_ctrl:
        c1, c2 = st.columns(2)
        if c1.button("＋ 追加"): st.session_state.g_count += 1
        if c2.button("－ 削除") and st.session_state.g_count > 2: st.session_state.g_count -= 1

    cols = st.columns(min(st.session_state.g_count, 4))
    for i in range(st.session_state.g_count):
        with cols[i % 4]:
            def_name = f"Group {i+1}"
            name = st.text_input(f"名前 {i+1}", value=def_name, key=f"n{i}")
            raw = st.text_area(f"データ {i+1}", height=120, key=f"d{i}", placeholder="10.5\n12.3")
            vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
            if len(vals) > 0: data_dict[name] = vals

# --- B. CSVアップロードモード ---
with tab_csv:
    uploaded_file = st.file_uploader("CSVファイルをアップロード (列名: Group, Value)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # 柔軟な列名対応
            if len(df.columns) >= 2:
                g_col = df.columns[0] # 1列目をグループ名と仮定
                v_col = df.columns[1] # 2列目を数値と仮定
                
                # 辞書に変換
                for g_name in df[g_col].unique():
                    g_vals = df[df[g_col] == g_name][v_col].dropna().tolist()
                    if len(g_vals) > 0:
                        data_dict[g_name] = g_vals
                st.success(f"CSV読み込み成功: {len(data_dict)} グループを検出")
            else:
                st.error("CSVは2列以上である必要があります (例: A列=グループ名, B列=数値)")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

# ---------------------------------------------------------
# 2. サイドバー設定 (グラフとデザイン)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🛠️ グラフ設定")
    
    with st.expander("📈 グラフの種類", expanded=True):
        graph_type = st.selectbox("形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
        if "棒" in graph_type:
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
        else:
            error_type = "None"
        
    with st.expander("🎨 デザイン微調整", expanded=True):
        fig_title = st.text_input("図のタイトル", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル", value="Relative Value")
        manual_y_max = st.number_input("Y軸最大値 (0で自動)", value=0.0, step=1.0)
        
        st.divider()
        st.caption("プロット調整")
        bar_width = st.slider("棒/箱の太さ", 0.1, 1.0, 0.6)
        dot_size = st.slider("ドットサイズ", 0, 100, 20)
        dot_alpha = st.slider("ドットの透明度", 0.1, 1.0, 0.7)
        jitter_strength = st.slider("ばらつき (Jitter)", 0.0, 0.2, 0.04, 0.01)
        fig_height = st.slider("画像の高さ", 3.0, 10.0, 5.0)

    # グループごとの色指定 (データがある場合のみ表示)
    group_colors = {}
    if data_dict:
        with st.expander("🖍️ グループカラー設定", expanded=True):
            default_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
            for i, g_name in enumerate(data_dict.keys()):
                col_def = default_colors[i % len(default_colors)]
                group_colors[g_name] = st.color_picker(f"{g_name} の色", col_def)
    
    st.divider()
    st.markdown("### 📢 Notice")
    st.caption("本ツールはベータ版です。論文等に使用する際は開発者までご連絡ください。")

st.divider()

# ---------------------------------------------------------
# 3. 統計解析エンジン (Logic Core)
# ---------------------------------------------------------
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

sig_pairs = [] 

if len(data_dict) >= 2:
    st.header("2. 統計解析レポート")
    
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # 診断: 正規性と等分散性
    all_normal = True
    for v in all_values:
        if len(v) >= 3:
            _, p_s = stats.shapiro(v)
            if p_s <= 0.05: all_normal = False
            
    try:
        _, p_lev = stats.levene(*all_values)
        is_equal_var = (p_lev > 0.05)
    except:
        is_equal_var = True

    # 検定ロジック
    method_name = ""
    p_global = 1.0
    
    # --- 2群比較 ---
    if len(data_dict) == 2:
        g1, g2 = all_values[0], all_values[1]
        if all_normal:
            method_name = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p_global = stats.ttest_ind(g1, g2, equal_var=is_equal_var)
        else:
            method_name = "Mann-Whitney U test"
            _, p_global = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            
        if p_global < 0.05:
            sig_pairs.append({'g1': group_names[0], 'g2': group_names[1], 'label': get_sig_label(p_global), 'p': p_global})

    # --- 3群以上比較 ---
    else:
        if all_normal and is_equal_var:
            method_name = "One-way ANOVA + Tukey's HSD"
            _, p_global = stats.f_oneway(*all_values)
            
            if p_global < 0.05:
                flat_data = [v for sub in all_values for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                for _, row in df_res.iterrows():
                    if row['reject']:
                        sig_pairs.append({'g1': row['group1'], 'g2': row['group2'], 'label': get_sig_label(row['p-adj']), 'p': row['p-adj']})
        else:
            method_name = "Kruskal-Wallis + Dunn's test"
            _, p_global = stats.kruskal(*all_values)
            
            if p_global < 0.05:
                dunn = sp.posthoc_dunn(all_values, p_adjust='bonferroni')
                dunn.columns = group_names
                dunn.index = group_names
                
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        n1, n2 = group_names[i], group_names[j]
                        p_val = dunn.loc[n1, n2]
                        if p_val < 0.05:
                            sig_pairs.append({'g1': n1, 'g2': n2, 'label': get_sig_label(p_val), 'p': p_val})

    # レポート表示
    # レポート表示
    st.success(f"**採用された手法: {method_name}**")
    
    # 日本語の親切な解説ロジックを追加
    if all_normal and is_equal_var:
        easy_reason = "データの分布に偏りがなく、群ごとのバラツキも均一であったため、最も標準的で統計的パワーの強い『パラメトリック検定』を採用しました。" [cite: 1]
    elif not all_normal:
        easy_reason = "データに正規性が認められなかった（極端な偏りや外れ値がある）ため、数値の順位に基づき、外れ値の影響を受けにくい『ノンパラメトリック検定』を採用しました。" [cite: 1]
    else:
        easy_reason = "群の間でバラツキ（分散）に有意な差が認められたため、その差を補正して計算する手法（Welchの方法等）を採用しました。" [cite: 1]

    result_summary = "【有意差あり】偶然とは言い切れない意味のある差が見つかりました。" if p_global < 0.05 else "【有意差なし】見られた差は誤差の範囲内である可能性が高いです。" [cite: 1]

    with st.expander("📝 そのまま使える報告用レポート (詳細)", expanded=True):
        full_report = f"""
【解析報告書：{", ".join(group_names)} の比較】

1. 解析の目的：
   各グループ間の数値に、統計学的な「意味のある違い」が存在するかを確認しました。 

2. 採用手法と選定理由：
   採用手法：{method_name}
   選定理由：{easy_reason}
   ※ データの正規性および等分散性を自動診断した上で、最も科学的に妥当な手順を選択しています。 

3. 解析結果：
   判定：{result_summary}
   全体のP値：{p_global:.4e}
   （※P値が0.05未満であれば、統計学的に「差がある」と判断します） 

4. 結論：
   以上の解析に基づき、有意差ラベル（{", ".join(set(p['label'] for p in sig_pairs)) if sig_pairs else "ns"}）を付与したグラフを作成しました。この結果は論文やレポートのエビデンスとして活用可能です。 
        """
        st.text_area("主査への説明やスライドのメモにコピペして使用してください", value=full_report, height=350)

# ---------------------------------------------------------
# 4. グラフ描画エンジン (Visualization Core)
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. グラフ生成 (Auto-Labeling)")
    
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(6, fig_height))
        
        group_names = list(data_dict.keys())
        x_positions = np.arange(len(group_names))
        
        # 最大値の計算 (Y軸調整用)
        max_val = -np.inf
        for v in data_dict.values():
            if len(v) > 0: max_val = max(max_val, max(v))
        if max_val == -np.inf: max_val = 1
        
        # --- A. プロット描画 ---
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM" else std_v
            
            # 色の取得 (サイドバー設定)
            my_color = group_colors.get(name, "#333333")

            if "棒" in graph_type:
                ax.bar(i, mean_v, width=bar_width, color=my_color, edgecolor='black', alpha=0.8, zorder=1)
                ax.errorbar(i, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "箱" in graph_type:
                ax.boxplot(vals, positions=[i], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=my_color, alpha=0.8), medianprops=dict(color='black'), showfliers=False)
            elif "バイオリン" in graph_type:
                parts = ax.violinplot(vals, positions=[i], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(my_color)
                    pc.set_alpha(0.8)
            
            # ドットプロット (Jitter & Alpha)
            if dot_size > 0:
                noise = np.random.normal(0, jitter_strength, len(vals))
                ax.scatter(x_positions[i] + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3, alpha=dot_alpha)

        # --- B. 有意差バーの自動描画 ---
        y_step = max_val * 0.15
        current_y = max_val * 1.1
        
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                x1, x2 = idx1, idx2
                bar_h = current_y
                col_h = max_val * 0.03
                
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='black')
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=14)
                
                current_y += y_step
            except: pass

        # --- C. レイアウト仕上げ ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        
        # Y軸範囲設定
        if manual_y_max > 0:
            ax.set_ylim(0, manual_y_max)
        else:
            ax.set_ylim(0, current_y * 1.05)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 画像を保存 (PNG)", data=img_buf, file_name=f"result_{now_str}.png", mime="image/png")
        
    except Exception as e:
        st.error(f"描画エラー: {e}")

else:
    st.info("データを入力してください (手動 または CSV)")
