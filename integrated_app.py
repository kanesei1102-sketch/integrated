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
# 1. ページ構成とセッション管理
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("統計解析から論文クオリティのグラフ作成までをシームレスに統合した完全版ツールです。")

# --- サイドバー: 共通設定 ---
with st.sidebar:
    st.header("🛠️ グラフ設定 (Graph Maker)")
    
    with st.expander("📈 グラフの種類", expanded=True):
        graph_type = st.selectbox("グラフ形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
        if "棒" in graph_type:
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
        else:
            error_type = "None"
        
        fig_title = st.text_input("図のタイトル", value="Comparison Results")
        y_axis_label = st.text_input("Y軸ラベル", value="Value")
        
    with st.expander("🎨 デザイン微調整"):
        bar_width = st.slider("幅 (Width)", 0.1, 1.0, 0.6)
        dot_size = st.slider("点のサイズ", 0, 100, 20)
        show_legend = st.checkbox("凡例を表示", value=False)
        fig_height = st.slider("画像の高さ", 3.0, 10.0, 5.0)

    st.write("---")
    st.markdown("""
    ### 【Notice / ご案内】
    本ツールはベータ版です。論文・学会発表等に使用される際は、以下のフォームより開発者（金子）までご連絡ください。
    
    👉 **[Contact Form / 連絡窓口](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)

# ---------------------------------------------------------
# 2. データ入力 (Stat Engine方式)
# ---------------------------------------------------------
if 'g_count' not in st.session_state: st.session_state.g_count = 3

st.subheader("1. データ入力")
c_ctl, _ = st.columns([1, 5])
with c_ctl:
    if st.button("＋ 群を追加"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ 群を削除"): st.session_state.g_count -= 1

data_dict = {}
cols = st.columns(min(st.session_state.g_count, 4)) # 列数は適宜調整
for i in range(st.session_state.g_count):
    with cols[i % 4]:
        def_name = f"Group {i+1}"
        name = st.text_input(f"名前 {i+1}", value=def_name, key=f"n{i}")
        raw = st.text_area(f"データ {i+1}", height=120, key=f"d{i}")
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) > 0: data_dict[name] = vals

st.divider()

# ---------------------------------------------------------
# 3. 解析エンジン (Stat Engine Core)
# ---------------------------------------------------------
# 有意差ラベル変換関数
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

sig_pairs = [] # グラフ描画用に有意差ペアを保存するリスト

if len(data_dict) >= 2:
    st.header("2. 統計解析レポート")
    
    # A. 診断
    all_normal = True
    for v in data_dict.values():
        if len(v) >= 3:
            _, p_s = stats.shapiro(v)
            if p_s <= 0.05: all_normal = False
    
    # 等分散性 (Levene)
    try:
        _, p_lev = stats.levene(*data_dict.values())
        is_equal_var = (p_lev > 0.05)
    except:
        is_equal_var = True # データ不足等の場合

    # B. 検定ロジック
    method_name = ""
    p_global = 1.0
    
    # --- 2群の場合 ---
    if len(data_dict) == 2:
        keys = list(data_dict.keys())
        g1, g2 = data_dict[keys[0]], data_dict[keys[1]]
        if all_normal:
            method_name = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p_global = stats.ttest_ind(g1, g2, equal_var=is_equal_var)
        else:
            method_name = "Mann-Whitney U test"
            _, p_global = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        st.info(f"採用手法: {method_name} (P={p_global:.4e})")
        if p_global < 0.05:
            sig_pairs.append({'g1': keys[0], 'g2': keys[1], 'p': p_global, 'label': get_sig_label(p_global)})

    # --- 3群以上の場合 ---
    else:
        if all_normal and is_equal_var:
            method_name = "One-way ANOVA + Tukey's HSD"
            _, p_global = stats.f_oneway(*data_dict.values())
            st.info(f"採用手法: {method_name} (Global P={p_global:.4e})")
            
            if p_global < 0.05:
                # Tukey HSD
                flat_data = [v for sub in data_dict.values() for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                
                # 結果を解析してsig_pairsに格納
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                for index, row in df_res.iterrows():
                    if row['reject']:
                        sig_pairs.append({'g1': row['group1'], 'g2': row['group2'], 'p': row['p-adj'], 'label': get_sig_label(row['p-adj'])})
                
                with st.expander("詳細な多重比較結果"):
                    st.table(df_res)
        
        else:
            method_name = "Kruskal-Wallis + Dunn's test"
            _, p_global = stats.kruskal(*data_dict.values())
            st.warning(f"採用手法: {method_name} (Global P={p_global:.4e})")
            
            if p_global < 0.05:
                # Dunn's test
                dunn_res = sp.posthoc_dunn(list(data_dict.values()), p_adjust='bonferroni')
                dunn_res.columns = dunn_res.index = data_dict.keys()
                
                # ペアごとの判定
                keys = list(data_dict.keys())
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        k1, k2 = keys[i], keys[j]
                        p_val = dunn_res.loc[k1, k2]
                        if p_val < 0.05:
                            sig_pairs.append({'g1': k1, 'g2': k2, 'p': p_val, 'label': get_sig_label(p_val)})
                
                with st.expander("詳細な多重比較結果"):
                    st.dataframe(dunn_res)

    # レポート生成機能 (Stat Engine)
    report_text = f"""【解析レポート】
手法: {method_name}
結果: {'有意差あり' if p_global < 0.05 else '有意差なし'} (P={p_global:.4e})
詳細: {len(sig_pairs)} 組のペアで有意な差が検出されました。
    """
    st.text_area("レポート (コピー用)", value=report_text, height=100)

else:
    st.write("データを入力すると解析とグラフ作成が始まります。")

st.divider()

# ---------------------------------------------------------
# 4. グラフ描画エンジン (Graph Maker Core)
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. グラフ生成 (Auto-Labeling)")
    
    try:
        # matplotlib設定
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(6, fig_height))
        
        group_names = list(data_dict.keys())
        x_positions = np.arange(len(group_names))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(group_names))) # 自動配色
        
        # データの最大値 (Y軸調整用)
        max_val = -np.inf
        for v in data_dict.values():
            if len(v) > 0: max_val = max(max_val, max(v))
        if max_val == -np.inf: max_val = 1
            
        # --- A. メインプロット ---
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            
            # 統計量
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM" else std_v

            # グラフ描画
            if "棒" in graph_type:
                ax.bar(i, mean_v, width=bar_width, color=colors[i], edgecolor='black', alpha=0.7, zorder=1)
                ax.errorbar(i, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "箱" in graph_type:
                ax.boxplot(vals, positions=[i], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=colors[i]), medianprops=dict(color='black'), showfliers=False)
            elif "バイオリン" in graph_type:
                parts = ax.violinplot(vals, positions=[i], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)

            # 個別プロット (Jitter)
            if dot_size > 0:
                noise = np.random.normal(0, 0.04, len(vals))
                ax.scatter(x_positions[i] + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3)

        # --- B. 有意差バー (Auto-Bracket) ---
        # ブラケットの高さを管理するためのオフセット
        y_step = max_val * 0.1
        current_y = max_val * 1.1
        
        # 有意差ペアをループして描画
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                
                # X座標
                x1, x2 = idx1, idx2
                
                # ブラケット描画
                bar_h = current_y
                col_h = max_val * 0.02
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='k')
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=12)
                
                # 次のバーのために高さを上げる
                current_y += y_step
            except:
                pass # グループ名不一致等のエラー回避

        # --- C. レイアウト調整 ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        
        # Y軸の上限設定（バーが切れないように）
        ax.set_ylim(0, current_y * 1.1)
        
        # シンプルな枠線
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
        
        # ダウンロードボタン
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 画像を保存 (PNG)", data=img_buf, file_name=f"graph_{now_str}.png", mime="image/png")

    except Exception as e:
        st.error(f"描画エラー: {e}")
