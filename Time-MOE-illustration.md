以下、図中 ① ～ ⑤ および中央の Transformer ＋ MoE ブロック内の各レイヤーを、「何を入力に」「どんな演算をし」「何を出力するか」という観点で、数式を交えて順に説明します。

---

## ① Point-wise Tokenization

時系列の生データ $\{x_t\}_{t=1}^T$（スカラーまたは多チャネル）をまず “点ごと” にベクトル化します。

$$
\mathbf{e}_t \;=\; \mathbf{W}^P\,x_t \;+\;\mathbf{b}^P
\quad\in\mathbb{R}^D
$$

- $x_t\in\mathbb{R}$（または $\mathbb{R}^C$）は時刻 $t$ の観測
- $\mathbf{W}^P\in\mathbb{R}^{D\times C},\ \mathbf{b}^P\in\mathbb{R}^D$ は学習可能な埋め込み行列

これにより、長さ $T$ の時系列が $T$ 個の $D$ 次元トークン $\{\mathbf{e}_t\}$ に変換されます。

---

## 中央：Transformer ＋ MoE スタック（全体を $N$ 層繰り返し）

各レイヤー $l=1,\dots,N$ で以下を行います。時刻 $t$ での表現を $\mathbf{h}_t^{l-1}$ とすると…

### 1. RMSNorm

$$
\mathbf{u}_t^{l-1}
=\mathrm{RMSNorm}\bigl(\mathbf{h}_t^{l-1}\bigr)
=\frac{\mathbf{h}_t^{l-1}}
{\sqrt{\tfrac1D\sum_{d=1}^D (h_{t,d}^{\,l-1})^2 + \epsilon}}
\;\odot\;\mathbf{g},
$$

- $\mathbf{g}\in\mathbb{R}^D$ は学習可能なスケール
- $\epsilon$ は数値安定化用

### 2. Causal Self-Attention

まずクエリ／キー／バリューを線形変換：

$$
\mathbf{q}_t=\mathbf{u}_t^{l-1}W^Q,\quad
\mathbf{k}_s=\mathbf{u}_s^{l-1}W^K,\quad
\mathbf{v}_s=\mathbf{u}_s^{l-1}W^V.
$$

因果マスク付き attention スコアと出力：

$$
\alpha_{t,s}
=\frac{\exp\!\bigl(\frac{\mathbf{q}_t\cdot \mathbf{k}_s}{\sqrt{d_k}}\bigr)}
{\sum_{u=1}^{t}\exp\!\bigl(\frac{\mathbf{q}_t\cdot \mathbf{k}_u}{\sqrt{d_k}}\bigr)},
\qquad
\mathbf{c}_t=\sum_{s=1}^{t}\alpha_{t,s}\,\mathbf{v}_s.
$$

### 3. 残差結合＋ RMSNorm

$$
\mathbf{r}_t^{l}
=\mathbf{h}_t^{\,l-1} + \mathbf{c}_t,
\qquad
\mathbf{v}_t^{l}
=\mathrm{RMSNorm}\bigl(\mathbf{r}_t^{l}\bigr).
$$

### 4. Mixture-of-Experts (MoE) Feed-Forward

1. **ルーター（Gate）**

   $$
   \boldsymbol{\pi}_t
   =\mathrm{softmax}\bigl(\mathbf{v}_t^{l}W^g + \mathbf{b}^g\bigr)
   \;\in\mathbb{R}^M,
   $$

   上位 $k=1$ のインデックス集合 $\mathcal{S}_t$ を取り、重みを再正規化。

2. **エキスパート**
   各専門家 $i=1,\dots,M$ に対し

   $$
   \mathrm{FFN}_i(\mathbf{v})
   =W_i^2\,\sigma\bigl(W_i^1\,\mathbf{v}+b_i^1\bigr)\;+\;b_i^2,
   $$

   $\sigma$ は活性化（例：ReLU）。

3. **混合出力**

   $$
   \mathbf{m}_t^{l}
   =\sum_{i\in\mathcal{S}_t}\pi_{t,i}\,\mathrm{FFN}_i\bigl(\mathbf{v}_t^{l}\bigr).
   $$

### 5. 最終残差結合

$$
\mathbf{h}_t^{l}
=\mathbf{r}_t^{l} + \mathbf{m}_t^{l}.
$$

以上を $l=1,\dots,N$ 繰り返し、最後の層出力 $\{\mathbf{h}_t^N\}$ を得ます。

---

## FM Head（最終予測ヘッド）

各時刻 $t$ の最終特徴 $\mathbf{h}_t^N$ を受け取り、Facto­rization-Machine 型の予測を行います。

$$
\widehat{y}_t
= w_0
+\sum_{d=1}^D w_d\,h_{t,d}^N
+\sum_{1\le i<j\le D}\langle \mathbf{v}_i,\mathbf{v}_j\rangle\,h_{t,i}^N\,h_{t,j}^N.
$$

ここで $\{w_d\},\{\mathbf{v}_d\}$ は学習パラメータ。

---

## ② Output Token Embeddings

FM ヘッドの出力 $\widehat{y}_t$ をさらに「出力トークン埋め込み」に変換するゲート付き FFN：

$$
\mathbf{z}_t
=\underbrace{\mathrm{Swish}\bigl(W_s\,\widehat{y}_t + b_s\bigr)}_{\text{ゲート}}
\;\odot\;
\underbrace{\bigl(W_f\,\widehat{y}_t + b_f\bigr)}_{\text{主体成分}}.
$$

---

## ③ MoE Output Hidden

$\mathbf{z}_t$ を受け、再び MoE で「パッチごとの隠れ表現」を出力：

$$
\boldsymbol{\rho}_t
=\mathrm{softmax}\bigl(\mathbf{z}_t W^r + b^r\bigr),\quad
\mathcal{S}_t=\mathrm{TopK}(\rho_t,\,k=1),
$$

$$
\mathbf{h}'_t
=\sum_{i\in\mathcal{S}_t}\rho_{t,i}\,\mathrm{Expert}_i(\mathbf{z}_t).
$$

---

## ④ Multi-task Optimization

得られた隠れ表現 $\mathbf{h}'_t$ を複数タスク（GT1, GT2, GT3）用の線形ヘッドでマルチ出力：

$$
\widehat{Y}_t^{(j)}
= W^{(j)}\,\mathbf{h}'_t + b^{(j)},
\quad j=1,2,3.
$$

損失はタスクごとに重み付き和で最適化：

$$
\mathcal{L}
=\sum_{j=1}^3 \lambda_j\,\ell\bigl(\widehat{Y}_t^{(j)},\,Y_t^{(j)}\bigr).
$$

---

## ⑤ Multi-resolution Scheduling

３つのタスク出力 $\{\widehat{Y}^{(j)}\}$ を、異なる時間解像度／パッチ長で合成し、最終的に複数パッチ（Patch 1, Patch 2,…）を作成します。

$$
\widehat{P}_p
=\sum_{j=1}^3 S_{p,j}\bigl(\widehat{Y}^{(j)}\bigr),
$$

ここで $S_{p,j}$ はパッチ $p$ に対してタスク $j$ の出力をダウンサンプリング／アップサンプリングする演算です。
（例：Patch 1 は粗い 24h 分解能、Patch 2 は細かい 1h 分解能、…）

---

以上が、図中 ① ～ ⑤ および中央の各レイヤーに対応する主な数式的定義です。各層が何を入力に、どのような線形変換・活性化・正規化・ゲーティングを行い、残差や MoE でどう結合しているかを示しました。
