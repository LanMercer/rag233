# 第三周力扣题目与答案汇总（专题：排序 + 递归 + 二分）

> 背景：本周零散时间刷题（0.5~1h/天），专题 **排序 + 递归 + 二分**（接续前两周"数组/哈希 → 字符串/链表"节奏），每日 1~2 题，中等难度封顶。
> 刷题方法（每周都适用）：先自己想 10 分钟（想不出也要想，这 10 分钟才是涨功力的部分）→ 看官方题解确认思路 → **用自己的话讲一遍**（面试考的就是这个）→ 提交通过后存进 GitHub `leetcode` 仓库，按 `排序/递归/二分` 分类，题解注释写清"数学视角"——面试时这就是谈资。
> 完成状态：✅ 已完成 **10/10**（912、215、56、75、104、226、704、35、34、69），已按分类提交 leetcode 仓库（状态经用户复核确认；912/215 与 34/69 在工作汇报时点记录为"待补"，周汇总前已确认全部完成）。
> 完成日期说明：各题按"周计划日期（对应 Day N）"记录，实际完成日期以各日教程验收清单与用户复核为准（本周日期因第二周收尾顺延：Day11 实际 8/20、Day12 8/21~8/22、Day13~15 9/2~9/3）。
>
> 数据结构/算法小抄（周内新增名词，初次出现即解释）：
>
> - **快速排序（Quick Sort）** = 分治排序：每轮选一个基准（pivot），把比它小的放左边、大的放右边，再递归两边。期望 O(n log n)；**随机选基准**把最坏 O(n²) 的概率降到几乎为 0。
> - **快速选择（Quick Select）** = 快排的"只进一边"版本：每轮分区后只递归答案所在的那半边，用来找"第 K 大/小"。
> - **递归（Recursion）** = 函数调用自己解决"规模更小但同构"的子问题——本质是数学归纳法/结构归纳法的代码化（归纳基 + 归纳步）。
> - **单调谓词（Monotonic Predicate）** = 一个"假假假…真真真"只翻转一次的条件判断，如 `P(i) = (nums[i] >= target)`；二分能找边界的前提就是谓词单调。
> - **牛顿法（Newton's Method）** = 用切线逐步逼近方程的根：每次在当前点做一阶泰勒展开、找切线与 x 轴交点作新估计，离根近时误差按平方速度收缩（**二次收敛**）。

---

## 专题一：排序（Sorting）

### 1. 912. 排序数组 —— 中等 ✅（8/15 · Day11 完成）｜ 必做，基础排序

**考点**：快排实现（原地 / 随机化）

**思路**：选基准（今天用随机基准避免最坏情况）→ 双指针 i/j 相向扫，把 `< pivot` 的换到左、`> pivot` 的换到右 → 递归左右两段。原地完成，不用额外大数组。

**代码**：

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        import random
        def quick(l, r):
            if l >= r:
                return
            pivot = nums[random.randint(l, r)]   # 随机选基准，避免最坏情况
            i, j = l, r
            while i <= j:
                while nums[i] < pivot: i += 1     # 左边找 >= pivot 的
                while nums[j] > pivot: j -= 1     # 右边找 <= pivot 的
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]  # 交换
                    i += 1; j -= 1
            quick(l, j); quick(i, r)              # 递归左右两段
        quick(0, len(nums) - 1)
        return nums
```

**复杂度**：期望 O(n log n) 时间、O(log n) 递归栈空间。**数学视角（随机化分析，面试可讲）**：快排每次把区间分成"小/基准/大"三段，递归深度期望 O(log n)——随机基准时每次划分"运气"期望平衡（一半一半），`E[深度]=O(log n)`；最坏 O(n²) 只在"每次基准恰好是最大/最小"时出现，现实中极难发生。这就是**随机化算法**"用随机性驯服最坏情况"的典型。

---

### 2. 215. 数组中的第 K 个最大元素 —— 中等 ✅（8/15 · Day11 完成）｜ **概率加成题**

**考点**：快速选择 / 堆

**思路**：快排的"只进一边"版——每轮随机选基准做分区得到位置 p：p 正好是第 K 个就返回；否则只递归答案所在的一边。

**代码**：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import random
        k = len(nums) - k          # 第 K 大 = 第 (n-K) 小（0 基）
        def quick_select(l, r):
            pivot = nums[random.randint(l, r)]
            i, j = l, r
            while i <= j:
                while nums[i] < pivot: i += 1
                while nums[j] > pivot: j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1; j -= 1
            if k < i and k > j:    # 基准正好落在缺口 = 找到了
                return pivot
            if k <= j:             # k 在左半段
                return quick_select(l, j)
            return quick_select(i, r)  # k 在右半段
        return quick_select(0, len(nums) - 1)
```

**复杂度**：期望 O(n)、最坏 O(n²)；空间 O(log n)。**数学视角（概率加成，必讲）**：每次分区只走一边，且随机基准使两边"期望对半"，总工作量 = `n + n/2 + n/4 + … = 2n = O(n)`——**等比级数求和**，这是统计人的数学强项，面试讲出来就是差异化谈资（对比排序后取第 K 个的 O(n log n)）。

---

### 3. 56. 合并区间 —— 中等 ✅（8/16 · Day12 完成）｜ 高频

**考点**：排序 + 贪心

**思路**：先按左端点排序，再贪心扫描——当前区间与"已合并的最后一个"重叠（`a <= res[-1][1]`）就并进去（右端点取大），否则开新区间。

**代码**：

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])          # 1. 按左端点排序
        res = []
        for a, b in intervals:                      # 2. 贪心扫描
            if not res or a > res[-1][1]:           # 与上一个不重叠
                res.append([a, b])
            else:                                   # 重叠 -> 右端点取大
                res[-1][1] = max(res[-1][1], b)
        return res
```

**复杂度**：O(n log n) 时间（排序主导）、O(n) 空间。**数学视角**：**排序让"重叠区间必相邻"**——排完序后贪心只需和上一个比较，不用两两配对（把"两两关系"压缩成"相邻关系"）；"排序 + 贪心"是区间类问题黄金组合。

---

### 4. 75. 颜色分类 —— 中等 ✅（8/16 · Day12 完成）｜ 经典三指针

**考点**：三指针（荷兰国旗）/ 一趟扫描

**思路**：维护三段区间不变量：`[0, l)` 全 0、`[l, i)` 全 1、`(r, n]` 全 2，`i` 一路扫把元素归位。注意：碰到 2 时 `i` **不前进**（交换来的值还要再判一次），这是最容易写错的地方。

**代码**：

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        i, l, r = 0, 0, len(nums) - 1   # 三指针：l 是 0 区右界，r 是 2 区左界
        while i <= r:
            if nums[i] == 0:            # 归到左边 0 区
                nums[l], nums[i] = nums[i], nums[l]
                l += 1; i += 1
            elif nums[i] == 2:          # 归到右边 2 区（i 不动，再判一次）
                nums[r], nums[i] = nums[i], nums[r]
                r -= 1
            else:                       # 1 留在中间
                i += 1
```

**复杂度**：一趟 O(n)、O(1) 额外空间。**数学视角**：**不变量驱动算法**——每一步都保持三段区间性质不变，证明正确性只需验证"初始化成立 / 每步保持 / 终止时得到答案"（循环不变式 = 数学归纳法的算法版）。

---

## 专题二：递归（Recursion）

### 5. 104. 二叉树的最大深度 —— 简单 ✅（8/17 · Day13 完成）｜ 必做，递归入门

**考点**：递归 / 分治

**思路**：递归三步走——① 终止条件（空节点返回 0）；② 本层干什么（取左右子树深度较大者 + 1）；③ 返回值（子树深度）。一棵树的最大深度 = `1 + max(左子树深度, 右子树深度)`。

**代码**：

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:                # 终止条件：空树深度 0
            return 0
        return 1 + max(self.maxDepth(root.left),   # 左子树深度
                       self.maxDepth(root.right))  # 右子树深度
```

**复杂度**：O(n) 时间；最坏退化成链时递归栈 O(n)（平衡树 O(log n)）。**数学视角**：**递归 = 数学归纳法**——空树是归纳基（`depth(∅)=0`），`depth(node) = 1 + max(depth(left), depth(right))` 是归纳步；只要基和步都对，递归必然正确，**不用脑补递归怎么展开**。

---

### 6. 226. 翻转二叉树 —— 简单 ✅（8/17 · Day13 完成）｜ 高频梗题

**考点**：递归（结构归纳）

**思路**：对每个节点先交换左右孩子，再递归翻转左右子树（或先递归后交换，等价）。终止条件为空节点。

**代码**：

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:                     # 终止条件：空节点
            return None
        root.left, root.right = root.right, root.left   # 先交换当前节点的左右孩子
        self.invertTree(root.left)       # 再翻转左子树
        self.invertTree(root.right)      # 再翻转右子树
        return root
```

**复杂度**：O(n) 时间、O(n) 最坏空间。**数学视角（结构归纳）**：翻转 = 对整棵树做"对称镜像"，可逐节点定义 `invert(node)` = 先 `invert(left)`、`invert(right)`，然后交换左右孩子——**先信任子问题已解决**（子树已翻好），只处理当前节点的互换。Homebrew 作者当年面试被这题挂掉，成了"递归经典梗题"。

---

## 专题三：二分查找（Binary Search）

### 7. 704. 二分查找 —— 简单 ✅（8/18 · Day14 完成）｜ 必做，最高频模板题

**考点**：二分查找模板

**思路**：维护搜索区间 `[lo, hi]`，每次取中点 `mid`：`nums[mid] < target` 去右半边、否则去左半边，直到区间为空；没找到返回 -1。

**代码**：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:                    # 区间还有元素
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                lo = mid + 1               # 目标只可能在右半边
            else:
                hi = mid - 1               # 目标只可能在左半边
        return -1
```

**复杂度**：O(log n) 时间、O(1) 空间。**数学视角**：二分 = **维护不变量**"答案一定在 [lo, hi] 里（若存在）"；每次比较把区间**砍半**，最多 `⌊log₂n⌋+1` 次——"区间砍半 → O(log n)"和分治、决策树高度是同一道理。不死循环证明：每次迭代区间严格缩小（`lo=mid+1`/`hi=mid-1`）必然有限步退出。

---

### 8. 35. 搜索插入位置 —— 简单 ✅（8/18 · Day14 完成）｜ 二分边界，必做

**考点**：二分边界（找第一个 ≥ target 的位置）

**思路**：二分终止时 `lo` 停的位置就是插入点。写法上维护不变量"`nums[0..lo-1] < target ≤ nums[hi+1..]`"，用 `lo < hi` + `hi = mid` 让 `lo` 收敛到第一个 `≥ target` 的下标（就是 Python 的 `bisect_left`）。

**代码**：

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums)              # 注意 hi 开区间写法：插入点可以是 len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:         # mid 太小，答案在右侧
                lo = mid + 1
            else:                          # nums[mid] >= target，mid 可能是答案
                hi = mid
        return lo
```

**复杂度**：O(log n) 时间、O(1) 空间。**数学视角（重点）**："找精确值"升级成"找边界" = **在单调谓词上找分界点**——定义 `P(i) = (nums[i] >= target)`，P 从假变真的位置就是插入点；二分能找边界的前提是 P **单调**（假假假…真真真），一旦不单调二分就失效。704 是"找等于"，35 是"找第一个大于等于"——**背下 35，34 题就是它的左右两次应用**。

---

### 9. 34. 在排序数组中查找元素的第一个和最后一个位置 —— 中等 ✅（8/19 · Day15 完成）｜ 必做进阶

**考点**：二分边界（左右边界两次应用）

**思路**：34 就是 35 的"左右两次应用"——用两个二分分别找 `nums[i] >= target` 的分界（第一个 target）和 `nums[i] > target` 的分界（最后一个 target + 1），再判断 target 存不存在。

**代码**：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def bisect_left(k: int) -> int:      # 第一个 >= k 的位置（等价于 python 的 bisect_left）
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < k:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        left = bisect_left(target)                     # P1: nums[i] >= target 的分界 = 第一个 target
        if left == len(nums) or nums[left] != target:  # target 不存在
            return [-1, -1]
        right = bisect_left(target + 1) - 1            # P2: 第一个 > target 的位置 - 1 = 最后一个 target
        return [left, right]
```

**复杂度**：两次二分，O(log n) 时间、O(1) 空间。**数学视角（重点）**：定义两个单调谓词 `P1(i) = (nums[i] >= target)`、`P2(i) = (nums[i] > target)`——二分能找边界的前提是谓词单调；`P1` 第一次变真的位置 = 左边界，`P2` 第一次变真的位置 = 右边界右一格。**"数组排好序" = 谓词天然单调**，这是 34 能用两次二分解的根本原因；把边界逻辑各包一个 `bisect_left`，比裸写 while 更不容易错。

---

### 10. 69. x 的平方根 —— 简单 ✅（8/19 · Day15 完成）｜ **数学加成题**

**考点**：牛顿法（数学主场）/ 二分

**思路一（二分，稳）**：找最大的 m 使 `m² ≤ x`，等价于在 [0, x] 上找谓词 `P(m) = (m² > x)` 第一次变真的位置减 1——和 34/35 同一个框架。
**思路二（牛顿法，数学主场）**：求 `sqrt(x)` = 求方程 `f(t) = t² − x = 0` 的正根，从 `t₀ = x` 迭代 `t_{n+1} = (t_n + x/t_n)/2`，收敛后用整数向下取整修正。

**代码**：

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x
        # 牛顿法：先算实数根近似（float 精度足够），再向下取整修正
        t = float(x)
        while True:
            nt = (t + x / t) / 2        # 牛顿迭代：中点 = 新估计
            if abs(nt - t) < 1e-7:      # 收敛（相邻两步差足够小）
                break
            t = nt
        r = int(t)
        while r * r > x:                # 浮点误差兜底：向下修正
            r -= 1
        while (r + 1) * (r + 1) <= x:   # 兜底：如果还能加就加
            r += 1
        return r
```

**复杂度**：牛顿法几步收敛（浮点内近似常数轮），二分 O(log x)。**数学视角（面试讲二次收敛）**：牛顿法 = 用**切线**逐步逼近根——每次在当前估计点做一阶泰勒展开、找切线与 x 轴交点作新估计；迭代公式推导：`t_{n+1} = t_n − f(t_n)/f'(t_n) = t_n − (t_n² − x)/(2t_n) = (t_n + x/t_n)/2`。**离根近时误差按平方速度收缩（二次收敛），精度每轮翻倍，通常几步就到机器精度**；而二分每次只砍一半（线性收敛，但绝对稳健）。能推导迭代公式 + 说出二次收敛 = "数学思维加分"的招牌讲法。

---

## 本周刷题总表

| 日期（计划/实际） | 题号 | 题名 | 考点 | 状态 |
|---|---|---|---|---|
| 8/15（Day11，实际 8/20） | 912 | 排序数组 | 快排 / 随机化分析 | ✅ |
| 8/15（Day11，实际 8/20） | 215 | 数组中的第 K 个最大元素 | 快速选择（概率加成） | ✅ |
| 8/16（Day12，实际 8/22） | 56 | 合并区间 | 排序 + 贪心 | ✅ |
| 8/16（Day12，实际 8/22） | 75 | 颜色分类 | 三指针（荷兰国旗） | ✅ |
| 8/17（Day13，实际 9/3 确认） | 104 | 二叉树的最大深度 | 递归 / 数学归纳法 | ✅ |
| 8/17（Day13，实际 9/3 确认） | 226 | 翻转二叉树 | 递归 / 结构归纳 | ✅ |
| 8/18（Day14，实际 9/3） | 704 | 二分查找 | 二分模板 / 区间砍半 | ✅ |
| 8/18（Day14，实际 9/3） | 35 | 搜索插入位置 | 单调谓词找分界 | ✅ |
| 8/19（Day15，实际 9/3） | 34 | 查找元素首末位置 | 二分边界（两次 bisect_left） | ✅ |
| 8/19（Day15，实际 9/3） | 69 | x 的平方根 | 牛顿法（二次收敛）/ 二分 | ✅ |

> 备注：912/215 在 Day11 工作汇报时点记录为"待补"，34/69 在 Day15 无独立工作汇报佐证——以上 4 题完成状态已在周汇总生成前经用户复核确认 ✅。本周专题三大类（排序/递归/二分）10 题全部收官，前两周（数组/哈希、字符串/链表）+ 本周共 30 题按分类沉淀进 leetcode 仓库。

---

## 收尾提醒

1. **GitHub `leetcode` 仓库**：本周 **10/10 题**已按 `排序/递归/二分` 分类提交（题解注释写"数学视角"）；前两周 20 题同步在库，可随时回顾。
2. **周计划"有余力再加"题（非必做，第 4 周可选）**：278 第一个错误的版本（二分经典）、148 排序链表（归并链表版，衔接链表专题）、153 寻找旋转排序数组中的最小值、33 搜索旋转排序数组（数学思维加分）、509 斐波那契数（递归 → 记忆化 → DP，DP 专题前置）、21/206 递归版复述。
3. **牛客"银行/央企"真题 1 套**：第三周计划项，⬜ 未完成 → 移至第 4 周补齐（熟悉"选择 + 编程 + 行测"混合题型）。

> 本汇总依据：`01-第三周详细计划.md` 刷题清单 + 各日教程"零散时间"思路提示（含完整代码与数学视角）+ 用户复核的完成状态整理；代码为标准题解写法，含"数学视角"注释。
