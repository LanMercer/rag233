# 第一周力扣题目与答案汇总（专题：数组 + 哈希表）

> 背景：本周零散时间刷题（0.5~~1h/天），专题 **数组 + 哈希表**，每日 1~~2 题，中等难度封顶。
> 刷题方法（每周都适用）：先自己想 10 分钟（想不出也要想，这 10 分钟才是涨功力的部分）→ 看官方题解确认思路 → **用自己的话讲一遍**（面试考的就是这个）→ 提交通过后存进 GitHub `leetcode` 仓库，按 `数组/哈希` 分类，题解注释写清"数学视角"——面试时这就是谈资。
> 完成状态：✅ 已完成 8/10（1、217、26、27、88、121、242、383）｜ 🔄 待收尾（283、169，题解均已备好）
>
> 数据结构小抄（周内新增名词，初次出现即解释）：
>
> - **哈希表（Hash Table）** = 可以理解成一个"翻牌台"：把键值直接放到对应格子，想知道"某个数以前见没见过"不用挨个翻，**查找速度 O(1) 极快**。
> - **双指针（Two Pointers）** = 在数组上用**两个手指头**（两个下标）同时从一头往另一头走，一个快一个慢，边走边配合，实现"一趟搞定"，省得用两层循环。
> - **字典（dict）** = Python 里的哈希表，`d = {"name": "张三"}`，用 `d["name"]` 取值，刷题天天见。

---



## 专题一：数组 / 双指针



### 1. 26. 删除有序数组中的重复项 —— 简单 ✅（8/2 完成）

**考点**：双指针 / 原地修改

**思路**：数组已排序，重复元素必相邻。慢指针 `i` 指向"最后一个不重复的位置"，快指针 `j` 从头扫到尾。只要 `nums[j] != nums[i]`，就把 `nums[j]` 放到 `i+1` 的位置并让 `i` 前进。一趟下来，`i+1` 就是不重复元素的个数。

**代码**：

```python
def removeDuplicates(nums):
    if not nums:
        return 0
    i = 0                          # 慢指针：最后一个不重复元素的位置
    for j in range(1, len(nums)):  # 快指针：扫描整个数组
        if nums[j] != nums[i]:     # 发现新元素
            i += 1
            nums[i] = nums[j]      # 搬到前面
    return i + 1
```

**复杂度**：时间 O(n)，空间 O(1)。**数学视角**：本质是"去重 = 只保留相邻不同的元素"，一趟遍历维护"已处理前缀"的不变量。

---



### 2. 27. 移除元素 —— 简单 ✅（8/2 完成）

**考点**：双指针 / 原地修改

**思路**：慢指针 `i` 指向"下一个要放的位置"，快指针 `j` 从头扫。只要 `nums[j] != val`，就把 `nums[j]` 放到 `i` 的位置，`i` 前进。最后 `i` 就是新长度（等于"保留下来的元素个数"）。

**代码**：

```python
def removeElement(nums, val):
    i = 0                          # 慢指针：下一个非 val 元素该放哪
    for j in range(len(nums)):     # 快指针
        if nums[j] != val:
            nums[i] = nums[j]      # 留下
            i += 1
    return i
```

**复杂度**：时间 O(n)，空间 O(1)。**数学视角**：和 26 题是同一套"慢指针划地、快指针找货"的模式；`i` 最终值就是"计数"。

---



### 3. 88. 合并两个有序数组 —— 简单 ✅（8/3 完成）

**考点**：双指针 / 从后往前

**思路**：nums1 有足够空间（前 m 个是有效元素），nums2 有 n 个。**从后往前填**：三个指针 `i = m-1`、`j = n-1`、`k = m+n-1`，每次都把更大的那个放到 `nums1[k]`。从后往前能避免覆盖 nums1 还没处理的元素。

**代码**：

```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:                      # 只剩 nums2 没放完时也要放（nums1 剩余天然有序）
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

**复杂度**：时间 O(m+n)，空间 O(1)。**数学视角**：归并排序的"合并"步骤（merge）——两个有序序列合并成一个有序序列，是分治思想的基石。这是本周和排序八股关联最紧的一题。

---



### 4. 121. 买卖股票的最佳时机 —— 简单 ✅（8/3 完成）

**考点**：一次遍历 / 维护最小值（DP 专题前置）

**思路**：只能买卖一次。遍历价格，维护"到目前为止的最小价格 `min_price`"，同时算"今天卖能赚多少 = prices[i] - min_price"，取最大值。核心想法：**低点买入的时机越早越好，卖出就看当天价**。

**代码**：

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        if price < min_price:
            min_price = price            # 更新历史最低买点
        elif price - min_price > max_profit:
            max_profit = price - min_price  # 今天卖能赚多少
    return max_profit
```

**复杂度**：时间 O(n)，空间 O(1)。**数学视角**：把"最大利润"拆成 `当前价 - 历史最低价` 的最大值，是"动态规划"思想的雏形——用一个变量记住前面信息（最优子结构：全局最优 = 每天"以今天结尾"的最优里挑最大）。

---



### 5. 283. 移动零 —— 简单  ✅（8/5 安排，报告时点进行中）

**考点**：双指针

**思路**：慢指针 `i` 负责"下一个非零数该放哪"，快指针 `j` 负责"找到下一个非零数"，把非零数依次搬到前面；**最后剩下的位置全填 0**。一趟遍历，O(n) 时间、O(1) 空间。

**代码**：

```python
def moveZeroes(nums):
    i = 0                               # 慢指针：下一个非零数该放哪
    for j in range(len(nums)):          # 快指针：找非零数
        if nums[j] != 0:
            nums[i], nums[j] = nums[j], nums[i]  # 非零数交换到前面
            i += 1
    # 剩余位置自然全是 0（因为非零数都被交换到前面了）
```

**复杂度**：时间 O(n)，空间 O(1)。**数学视角**：这题是"快速排序分区（partition）"的精简版——把"不等于 pivot"的元素往左放。理解它 = 理解快排的核心循环。

---



### 6. 169. 多数元素 —— 简单  ✅（8/5 安排，报告时点进行中）｜ **数学背景加分题**

**考点**：摩尔投票（Boyer-Moore Voting）/ 哈希

**思路（数学视角，重点）**：若某数出现次数超过一半，则"两个不同数互相抵消"之后，**最后剩下来的就是它**。用概率直觉：超过一半 = 哪怕每遇到一个不同数就抵消一次，它也抵消不完，必胜。这就是摩尔投票——一趟遍历、O(1) 空间。

**代码**：

```python
def majorityElement(nums):
    candidate = None
    count = 0
    for num in nums:
        if count == 0:      # 抵消光了，换新候选
            candidate = num
        count += 1 if num == candidate else -1   # 同阵营 +1，不同阵营 -1
    return candidate
```

**复杂度**：时间 O(n)，空间 O(1)。**数学视角**：面试时用"超过一半 = 抵消不完"这句概率语言讲，特别加分。也可用哈希统计（O(n) 空间）或排序取中位数（O(nlogn)）作对比方案。

---



## 专题二：哈希表



### 7. 1. 两数之和 —— 简单 ✅（8/1 完成）｜ 笔试出现率极高，必做

**考点**：哈希表 / 双循环

**思路**：遍历数组，用一个字典存"见过的数 → 它的下标"。对每个数 `nums[i]`，算"我缺谁"`target - nums[i]`，去字典里翻：翻到了就直接返回两个下标；没翻到就把当前数放进去。一次遍历搞定。

**代码**：

```python
def twoSum(nums, target):
    seen = {}                       # 哈希表：{数字: 下标}
    for i, num in enumerate(nums):
        need = target - num         # 我缺谁
        if need in seen:            # 翻牌台上一查
            return [seen[need], i]
        seen[num] = i               # 没凑成，把自己放上去
    return []
```

**复杂度**：时间 O(n)，空间 O(n)。**数学视角**：把"找两数"变成"边扫边查缺"，是哈希表最经典的"用空间换时间"。暴力双循环是 O(n²)，哈希把它降到 O(n)。

---



### 8. 217. 存在重复元素 —— 简单 ✅（8/1 完成）

**考点**：哈希 / 排序

**思路**：用一个集合（set）记录见过的数，遍历时如果当前数已经在集合里 → 有重复返回 True。或者排序后看相邻元素是否相等。

**代码**：

```python
def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:      # 已经在翻牌台上 = 重复
            return True
        seen.add(num)
    return False
```

**复杂度**：时间 O(n)，空间 O(n)。**数学视角**：重复检测 = 集合论里"交集非空"问题，哈希集合给出 O(n) 解法；备选方案排序 O(nlogn)、暴力 O(n²)。

---



### 9. 242. 有效的字母异位词 —— 简单 ✅（8/4 完成）

**考点**：哈希计数

**思路**：字母异位词 = 两个字符串用的字母种类和个数都完全一样。统计 s 每个字母出现次数，再遍历 t 逐个"消耗"：消耗不完（出现负数或缺字母）就不是异位词。本质是"计数相同 = 分布相同"。

**代码**：

```python
from collections import Counter

def isAnagram(s, t):
    if len(s) != len(t):      # 长度不同必不可能
        return False
    count = Counter(s)        # 统计 s 的字母频数（Counter = 计数用的字典）
    for ch in t:
        if count[ch] == 0:
            return False      # t 里多出 s 没有的 / 用超了的字母
        count[ch] -= 1        # 消耗掉一个
    return True




    def isAnagram(self, s: str, t: str) -> bool:
        s_e={}
        for letter1 in s:
            s_e[letter1]=s_e.get(letter1, 0) + 1
        t_e={}
        for letter2 in t:
            t_e[letter2]=t_e.get(letter2, 0) + 1
        if s_e==t_e:
            return True
        else:
            return False

```

**复杂度**：时间 O(n)，空间 O(1)（只有 26 个小写字母）。**数学视角**：异位词 = 两个离散分布的直方图完全相同。这就是统计学里"比较两个分类变量分布"的最简版本。

---



### 10. 383. 赎金信 —— 简单 ✅（8/4 完成）

**考点**：哈希计数（高频）

**思路**：和 242 同一套路——杂志 `magazine` 提供字母库存，`ransomNote` 消耗库存。统计 magazine 每个字母数量，再遍历 ransomNote 看每个字母**够不够用**。

**代码**：

```python
from collections import Counter

def canConstruct(ransomNote, magazine):
    stock = Counter(magazine)     # 杂志字母库存
    for ch in ransomNote:
        if stock[ch] == 0:
            return False          # 库存不够，构造不出来
        stock[ch] -= 1            # 用掉一个
    return True




    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        r_e = {}
        for letter1 in ransomNote:
            r_e[letter1] = r_e.get(letter1, 0) + 1
            
        m_e = {}
        for letter2 in magazine:
            m_e[letter2] = m_e.get(letter2, 0) + 1
            
    
        for letter, count in r_e.items():
            if m_e.get(letter, 0) < count:
                return False
                
        return True    
```

**复杂度**：时间 O(n+m)，空间 O(1)。**数学视角**：把"能否拼出"翻译成"每个字母的需求量 ≤ 库存量"，是离散分布比较的又一次应用（242 是"是否相等"，383 是"是否够用"）。

---



## 本周刷题总表


| 日期     | 题号  | 题名          | 考点          | 状态  |
| ------ | --- | ----------- | ----------- | --- |
| 8/1 周六 | 1   | 两数之和        | 哈希          | ✅   |
| 8/1 周六 | 217 | 存在重复元素      | 哈希          | ✅   |
| 8/2 周日 | 26  | 删除有序数组中的重复项 | 双指针         | ✅   |
| 8/2 周日 | 27  | 移除元素        | 双指针         | ✅   |
| 8/3 周一 | 88  | 合并两个有序数组    | 双指针 / 从后往前  | ✅   |
| 8/3 周一 | 121 | 买卖股票的最佳时机   | 一次遍历（DP 前置） | ✅   |
| 8/4 周二 | 242 | 有效的字母异位词    | 哈希计数        | ✅   |
| 8/4 周二 | 383 | 赎金信         | 哈希计数        | ✅   |
| 8/5 周三 | 283 | 移动零         | 双指针         | ✅   |
| 8/5 周三 | 169 | 多数元素        | 摩尔投票（数学思维）  | ✅   |




## 收尾提醒（本周遗留）

- 在 GitHub 建 `leetcode` 仓库，按 `数组/哈希` 分类提交，题解注释写清"数学视角"（面试谈资）；
- 第二周刷题专题预告：字符串 + 链表（周计划见 `第二周` 详细计划）。

> 本汇总依据：`01-第一周详细计划.md` 刷题清单 + 各日教程"零散时间"思路提示整理；代码为标准题解写法，含"数学视角"注释。

