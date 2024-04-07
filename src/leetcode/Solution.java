package leetcode;


import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author lilei
 **/
@SuppressWarnings("unused")
public class Solution {
    /**
     * 找到字符串中所有字母异位词
     */
    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length()) {
            return new ArrayList<>();
        }
        int[] sCount = new int[26];
        int[] pCount = new int[26];
        for (int i = 0; i < p.length(); i++) {
            sCount[s.charAt(i) - 'a']++;
            pCount[p.charAt(i) - 'a']++;
        }

        List<Integer> res = new ArrayList<>();
        int left = 0;
        int right = p.length() - 1;

        while (true) {
            if (Arrays.equals(sCount, pCount)) {
                res.add(left);
            }

            if (right == s.length() - 1) {
                break;
            }

            sCount[s.charAt(left++) - 'a']--;
            sCount[s.charAt(++right) - 'a']++;
        }

        return res;
    }

    /**
     * 字母异位词分组
     * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
     * 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
     * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
     * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            int[] count = new int[26];
            for (char c : str.toCharArray()) {
                count[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; i++) {
                if (count[i] > 0) {
                    sb.append((char) ('a' + i));
                    sb.append(count[i]);
                }
            }
            List<String> list = map.get(sb.toString());
            if (list == null) {
                List<String> ss = new ArrayList<>();
                ss.add(str);
                map.put(sb.toString(), ss);
            } else {
                list.add(str);
            }
        }

        return new ArrayList<>(map.values());
    }


    /**
     * 两数之和
     */
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int look = target - nums[i];
            Integer integer = map.get(look);
            if (integer != null) {
                res[0] = integer;
                res[1] = i;
            } else {
                map.put(nums[i], i);
            }
        }
        return res;
    }

    /**
     * 轮转数组
     * 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     */
    public void rotate(int[] nums, int k) {
        int offset = k % nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, offset - 1);
        reverse(nums, offset, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    /**
     * 给定一个经过编码的字符串，返回它解码后的字符串。
     * <p>
     * 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
     * <p>
     * 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
     * <p>
     * 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像3a或2[4]的输入。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：s = "3[a]2[bc]"
     * 输出："aaabcbc"
     * 示例 2：
     * <p>
     * 输入：s = "3[a2[c]]"
     * 输出："accaccacc"
     * 示例 3：
     * <p>
     * 输入：s = "2[abc]3[cd]ef"
     * 输出："abcabccdcdcdef"
     * 示例 4：
     * <p>
     * 输入：s = "abc3[cd]xyz"
     * 输出："abccdcdcdxyz"
     */
    public String decodeString(String s) {
        Stack<StringBuilder> letterStack = new Stack<>();
        Stack<Integer> digitStack = new Stack<>();

        StringBuilder letterSb = new StringBuilder();
        StringBuilder digitSb = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);

            if (Character.isDigit(c)) {
                digitSb.append(c);
            } else if (c == '[') {
                int digit = Integer.parseInt(digitSb.toString());
                digitStack.add(digit);
                letterStack.add(new StringBuilder(letterSb));
                digitSb.setLength(0);
                letterSb.setLength(0);
            } else if (c == ']') {
                StringBuilder temp = new StringBuilder();
                Integer digit = digitStack.pop();
                for (int j = 0; j < digit; j++) {
                    temp.append(letterSb);
                }
                StringBuilder letter = letterStack.pop();
                letterSb = letter.append(temp);

            } else {
                letterSb.append(c);
            }
        }

        return letterSb.toString();
    }


    /**
     * 前K个高频元素
     * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
     * <p>
     * 示例 1:
     * <p>
     * 输入: nums = [1,1,1,2,2,3], k = 2
     * 输出: [1,2]
     * 示例 2:
     * <p>
     * 输入: nums = [1], k = 1
     * 输出: [1]
     */
    public int[] topKFrequent(int[] nums, int k) {
        int[] res = new int[k];
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((x, y) -> map.get(x) - map.get(y));
        for (Integer key : map.keySet()) {
            pq.offer(key);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll();
        }
        return res;
    }

    public int[] topKFrequent1(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            Integer integer = map.get(nums[i]);

            if (integer == null) {
                integer = 1;
            } else {
                integer++;
            }
            map.put(nums[i], integer);
        }

        List<Map.Entry<Integer, Integer>> collect = map.entrySet().
                stream().sorted((v1, v2) -> v2.getValue() - v1.getValue()).collect(Collectors.toList());

        int[] res = new int[k];
        for (int i = 0; i < k; i++) {

            res[i] = collect.get(i).getKey();
        }

        return res;
    }


    /**
     * 寻找重复数
     * 给定一个包含n + 1 个整数的数组nums ，其数字都在[1, n]范围内（包括 1 和 n），可知至少存在一个重复的整数。
     * <p>
     * 假设 nums 只有 一个重复的整数 ，返回这个重复的数 。
     * <p>
     * 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [1,3,4,2,2]
     * 输出：2
     * 示例 2：
     * <p>
     * 输入：nums = [3,1,3,4,2]
     * 输出：3
     */
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        fast = 0;
        while (fast != slow) {
            fast = nums[fast];
            slow = nums[slow];
        }

        return fast;
    }



    /**
     * 除自身以外数组的乘积
     */
    public int[] productExceptSelf(int[] nums) {
        int[] result = new int[nums.length];
        int[] after = new int[nums.length];

        //计算前缀之积
        result[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            result[i] = nums[i - 1] * result[i - 1];
        }

        after[nums.length - 1] = 1;
        for (int i = nums.length - 2; i >= 0; i--) {
            after[i] = after[i + 1] * nums[i + 1];
            result[i] = result[i] * after[i];
        }
        return result;
    }

    /**
     * 二叉树的最近公共祖先
     */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left != null && right != null) {
            return root;
        }

        if (left != null) {
            return left;
        }

        if (right != null) {
            return right;
        }

        return null;
    }

    //二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        HashMap<Integer, TreeNode> parentMap = new HashMap<>();
        //计算所有
        dfsP(root, parentMap);

        Set<TreeNode> pParent = new HashSet<>();
        //寻找p的所有父节点
        while (p != null) {
            pParent.add(p);
            p = parentMap.get(p.val);
        }

        //寻找q和p的最近公共节点
        while (q != null) {
            if (pParent.contains(q)) {
                return q;
            }
            q = parentMap.get(q.val);
        }
        return null;
    }

    public void dfsP(TreeNode node, Map<Integer, TreeNode> parentMap) {
        if (node == null) {
            return;
        }

        if (node.left != null) {
            parentMap.put(node.left.val, node);
            dfsP(node.left, parentMap);
        }
        if (node.right != null) {
            parentMap.put(node.right.val, node);
            dfsP(node.right, parentMap);
        }
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        Map<Integer, TreeNode> parentMap = new HashMap<>();
        dfs(root, parentMap);//存储所有节点val和对应parent的关系

        //依次先找到p的所有父节点的val
        List<Integer> pp = new ArrayList<>();
        while (p != null) {
            pp.add(p.val);
            p = parentMap.get(p.val);
        }

        while (q != null) {
            if (pp.contains(q.val)) {
                return q;
            }
            q = parentMap.get(q.val);
        }
        return null;
    }

    public void dfs(TreeNode node, Map<Integer, TreeNode> map) {
        if (node.left != null) {
            map.put(node.left.val, node);
            dfs(node.left, map);
        }
        if (node.right != null) {
            map.put(node.right.val, node);
            dfs(node.right, map);
        }
    }

    //二叉树中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        midDfs(root, list);
        return list;
    }

    public void midDfs(TreeNode node, List<Integer> list) {
        if (node == null) {
            return;
        }
        midDfs(node.left, list);
        list.add(node.val);
        midDfs(node.right, list);
    }

    // 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.right = left;
        root.left = right;
        return root;
    }

    //对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return isEqual(root, root);
    }

    public boolean isEqual(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val == right.val) {
            return isEqual(left.left, right.right) && isEqual(left.right, right.left);
        } else {
            return false;
        }

    }


    int maxPath = 0;

    //二叉树的直径
    public int diameterOfBinaryTree(TreeNode root) {
        maxDepth1(root);
        return maxPath - 1;
    }

    public int maxDepth1(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int leftMaxDepth = maxDepth1(root.left);
        int rightMaxDepth = maxDepth1(root.right);
        maxPath = Math.max(maxPath, leftMaxDepth + rightMaxDepth + 1);
        return Math.max(leftMaxDepth, rightMaxDepth) + 1;
    }

    //将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int left, int right) {
        int mid = (left + right) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBST(nums, left, mid - 1);
        node.right = sortedArrayToBST(nums, mid + 1, right);
        return node;
    }


    /**
     * 零钱兑换
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     * <p>
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回-1 。
     * <p>
     * 你可以认为每种硬币的数量是无限的。
     */
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;

        for (int j = 0; j < coins.length; j++) {
            for (int i = coins[j]; i <= amount; i++) {
                //总金额为i的最小硬币数量
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        if (dp[amount] > amount) {
            return -1;
        }
        return dp[amount];
    }


    //零钱兑换
    //给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
    //
    //请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
    //
    //假设每一种面额的硬币有无限个。
    //
    //题目数据保证结果符合 32 位带符号整数。
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] = dp[j] + dp[j - coins[i]];
            }
        }
        return dp[amount];
    }


    /**
     * 最大正方形
     * 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
     * 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
     * 输出：4
     */
    public int maximalSquare(char[][] matrix) {
        int x = matrix.length;
        int y = matrix[0].length;
        int[][] dp = new int[x][y];
        int max = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        int min = Math.min(dp[i - 1][j - 1], dp[i - 1][j]);
                        dp[i][j] = Math.min(min, dp[i][j - 1]) + 1;
                    }
                    max = Math.max(max, dp[i][j]);
                }
            }
        }
        return max * max;
    }


    /**
     * 课程表
     * <p>
     * 现在你总共有 numCourses 门课需要选，记为0到numCourses - 1。给你一个数组prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修bi 。
     * <p>
     * 例如，想要学习课程 0 ，你需要先完成课程1 ，我们用一个匹配来表示：[0,1] 。
     * 返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：numCourses = 2, prerequisites = [[1,0]]
     * 输出：[0,1]
     * 解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
     * 示例 2：
     * <p>
     * 输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
     * 输出：[0,2,1,3]
     * 解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
     * 因此，一个正确的课程顺序是[0,1,2,3] 。另一个正确的排序是[0,2,1,3] 。
     * 示例 3：
     * <p>
     * 输入：numCourses = 1, prerequisites = []
     * 输出：[0]
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0) {
            return new int[0];
        }
        //记录所有入度
        int[] inDegree = new int[numCourses];
        for (int[] prerequisite : prerequisites) {
            inDegree[prerequisite[0]]++;
        }
        //找出所有入度为0的数放入队列
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }


        int index = 0;
        int[] result = new int[numCourses];
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            result[index++] = poll;

            //遍历所有节点，将入度为poll的节点的入度数减一
            for (int[] prerequisite : prerequisites) {
                if (prerequisite[1] == poll) {
                    inDegree[prerequisite[0]]--;
                    if (inDegree[prerequisite[0]] == 0) {
                        queue.offer(prerequisite[0]);
                    }
                }
            }
        }

        if (index == numCourses) {
            return result;
        }
        return new int[0];
    }

    /**
     * 打家劫舍
     * <p>
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * <p>
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：[1,2,3,1]
     * 输出：4
     * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     * 偷窃到的最高金额 = 1 + 3 = 4 。
     * 示例 2：
     * <p>
     * 输入：[2,7,9,3,1]
     * 输出：12
     * 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     * 偷窃到的最高金额 = 2 + 9 + 1 = 12 。
     */
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }


    /**
     * 排序链表
     * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        //找到中间节点
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null) {
            if (fast.next.next == null) {
                break;
            }
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode mid = slow.next;
        slow.next = null;
        ListNode sortList1 = sortList(head);
        ListNode sortList2 = sortList(mid);
        return merge(sortList1, sortList2);
    }

    //合并两个有序链表
    private ListNode merge(ListNode node1, ListNode node2) {
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;
        while (node1 != null && node2 != null) {
            if (node1.val <= node2.val) {
                dummy.next = node1;
                node1 = node1.next;
            } else {
                dummy.next = node2;
                node2 = node2.next;
            }
            dummy = dummy.next;
        }
        if (node1 == null) {
            dummy.next = node2;
        }
        if (node2 == null) {
            dummy.next = node1;
        }
        return head.next;
    }


    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }

        public static void print(ListNode node) {
            if (node != null) {
                System.out.println(node.val);
                print(node.next);
            }
        }
    }

    /**
     * 环形链表
     * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     */
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> visited = new HashSet<>();
        while (head != null) {
            if (!visited.contains(head)) {
                visited.add(head);
                head = head.next;
            } else {
                return head;
            }
        }
        return null;
    }

    public ListNode detectCycle1(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                while (head != fast) {
                    head = head.next;
                    fast = fast.next;
                }
                return fast;
            }
        }

        return null;
    }

    /**
     * 单词拆分
     * <p>
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     * <p>
     * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
     * <p>
     * 输入: s = "leetcode", wordDict = ["leet", "code"]
     * 输出: true
     * 解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
     * 示例 2：
     * <p>
     * 输入: s = "applepenapple", wordDict = ["apple", "pen"]
     * 输出: true
     * 解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     *      注意，你可以重复使用字典中的单词。
     * 示例 3：
     * <p>
     * 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
     * 输出: false
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }


    /**
     * 验证二叉搜索树
     */
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    //437. 路径总和 III
    //给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
    //路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
    int count = 0;

    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> map = new HashMap<>();
        map.put(0L, 1);
        dfs(root, targetSum, 0, map);
        return count;
    }

    private void dfs(TreeNode node, int targetSum, long sum, Map<Long, Integer> map) {
        if (node == null) {
            return;
        }
        sum += node.val;
        if (map.containsKey(sum - targetSum)) {
            count += map.get(sum - targetSum);
        }
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        dfs(node.left, targetSum, sum, map);
        dfs(node.right, targetSum, sum, map);
        map.put(sum, map.get(sum) - 1);
    }

    //从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        Map<Integer,Integer>indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return buildTree(preorder,inorder,0,preorder.length-1,0,indexMap);
    }

    public TreeNode buildTree(int[] preorder, int[] inorder, int left, int right, int root, Map<Integer, Integer> indexMap) {
        if (left > right) {
            return null;
        }
        Integer mid = indexMap.get(preorder[root]);
        TreeNode node = new TreeNode(preorder[root]);
        node.left = buildTree(preorder, inorder, left, mid - 1, root + 1, indexMap);
        node.right = buildTree(preorder, inorder, mid + 1, right, root + 1 + mid - left, indexMap);
        return node;
    }


    //验证二叉搜索树
    public boolean isValidBST1(TreeNode root) {
        return idValidBST1(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean idValidBST1(TreeNode node, long min, long max) {
        if (node == null) {
            return true;
        }
        return node.val > min && node.val < max
                && idValidBST1(node.left, min, node.val) && idValidBST1(node.right, node.val, max);
    }

    int res;

    //二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> deque = new LinkedList<>();
        while (root != null || !deque.isEmpty()) {
            while (root != null) {
                deque.push(root);
                root = root.left;
            }
            root = deque.pop();
            k--;
            if (k == 0) {
                break;
            }
            root = root.right;
        }
        assert root != null;
        return root.val;
    }

    List<Integer> list = new ArrayList<>();

    //二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        rightSideView(root, 0);
        return list;
    }

    public void rightSideView(TreeNode node, int depth) {
        if (node == null) {
            return;
        }
        if (list.size() == depth) {
            list.add(node.val);
        }
        depth++;
        rightSideView(node.left, depth);
        rightSideView(node.right, depth);
    }

    //二叉树展开为链表
    public void flatten(TreeNode root) {
        List<TreeNode> nodes = new ArrayList<>();
        flatten(root, nodes);

        for (int i = 0; i < nodes.size(); i++) {
            TreeNode node = nodes.get(i);
            node.left = null;
            if (i < nodes.size() - 1) {
                node.right = nodes.get(i + 1);
            } else {
                node.right = null;
            }

        }
    }

    public void flatten(TreeNode root, List<TreeNode> list) {
        if (root == null) {
            return;
        }
        list.add(root);
        flatten(root.left, list);
        flatten(root.right, list);
    }


    /**
     * 层序遍历
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        List<TreeNode> treeNodes = new ArrayList<>();
        treeNodes.add(root);
        levelOrder(treeNodes, result);
        return result;
    }

    private void levelOrder(List<TreeNode> nodes, List<List<Integer>> result) {
        if (nodes.isEmpty()) {
            return;
        }
        List<Integer> list = new ArrayList<>();
        List<TreeNode> treeNodes = new ArrayList<>();
        for (TreeNode node : nodes) {
            list.add(node.val);
            if (node.left != null) {
                treeNodes.add(node.left);
            }
            if (node.right != null) {
                treeNodes.add(node.right);
            }
        }
        result.add(list);
        levelOrder(treeNodes, result);
    }


    /**
     * 给定一个m x n 二维字符网格board 和一个字符串单词word 。如果word 存在于网格中，返回 true ；否则，返回 false 。
     *
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     *
     */
//    public boolean exist(char[][] board, String word) {
//        int xLen = board.length;
//        int yLen = board[0].length;
//        for (int i = 0; i < xLen; i++) {
//            for (int j = 0; j < yLen; j++) {
//                if(exist(board,word,0,i,j)){
//                    return true;
//                }
//            }
//        }
//        return false;
//    }
//
//    public boolean exist(char[][] board, String word, int index, int x, int y) {
//        int xLen = board.length;
//        int yLen = board[0].length;
//        if (x < 0 || x >= xLen || y < 0 || y >= yLen || board[x][y] != word.charAt(index)) {
//            return false;
//        }
//
//        if (index == word.length() - 1) {
//            return true;
//        }
//
//        char temp = board[x][y];
//        board[x][y] = ' ';
//        if (exist(board, word, index + 1, x + 1, y) ||
//                exist(board, word, index + 1, x - 1, y) ||
//                exist(board, word, index + 1, x, y + 1) ||
//                exist(board, word, index + 1, x, y - 1)) {
//            return true;
//        }
//        board[x][y] = temp;
//        return false;
//    }

    /**
     * 给定一个包含红色、白色和蓝色、共n 个元素的数组nums，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
     * <p>
     * 我们使用整数 0、1 和 2 分别表示红色、白色和蓝色。
     * <p>
     * 必须在不使用库内置的 sort 函数的情况下解决这个问题。
     */
    public void sortColors(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            Integer integer = map.get(num);
            if (integer == null) {
                map.put(num, 1);
            } else {
                map.put(num, integer + 1);
            }
        }

        Integer integer1 = map.get(0);
        Integer integer2 = map.get(1);
        Integer integer3 = map.get(2);
        if (integer1 == null) {
            integer1 = 0;
        }
        if (integer2 == null) {
            integer2 = 0;
        }
        if (integer3 == null) {
            integer3 = 0;
        }
        for (int i = 0; i < integer1; i++) {
            nums[i] = 0;
        }
        for (int i = integer1; i < integer1 + integer2; i++) {
            nums[i] = 1;
        }
        for (int i = integer1 + integer2; i < integer1 + integer2 + integer3; i++) {
            nums[i] = 2;
        }
    }

    //双指针
    public void sortColors1(int[] nums) {
        if (nums.length < 2) {
            return;
        }
        int i = 0;
        int p0 = 0;
        int p2 = nums.length - 1;
        while (i <= p2) {
            if (nums[i] == 0) {
                swap(nums, i, p0);
                i++;
                p0++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                swap(nums, i, p2);
                p2--;
            }
        }
    }


    /**
     * 不同路径
     * 一个机器人位于一个 m x n网格的左上角 （起始点在下图中标记为 “Start” ）。
     * <p>
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * <p>
     * 问总共有多少条不同的路径？
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
     * 输出：[[1,6],[8,10],[15,18]]
     * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     * 示例2：
     * <p>
     * 输入：intervals = [[1,4],[4,5]]
     * 输出：[[1,5]]
     * 解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
     */
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(interval -> interval[0]));

        List<int[]> list = new ArrayList<>();
        list.add(intervals[0]);

        for (int i = 1; i < intervals.length; i++) {
            int[] last = list.get(list.size() - 1);
            if (intervals[i][0] <= last[1]) {
                //重叠
                last[1] = Math.max(intervals[i][1], last[1]);
            } else {
                list.add(intervals[i]);
            }
        }

        return list.toArray(new int[list.size()][]);

    }


    /**
     * 最大子数组和
     * <p>
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * <p>
     * 子数组 是数组中的一个连续部分。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
     * 输出：6
     * 解释：连续子数组[4,-1,2,1] 的和最大，为6 。
     * 示例 2：
     * <p>
     * 输入：nums = [1]
     * 输出：1
     * 示例 3：
     * <p>
     * 输入：nums = [5,4,-1,7,8]
     * 输出：23
     */
    public int maxSubArray(int[] nums) {
        int preMax = 0;//索引前面的最大子数组
        int max = nums[0];
        for (int num : nums) {
            preMax = Math.max(preMax + num, num);
            max = Math.max(max, preMax);
        }
        return max;
    }

    /**
     * 字母异位词分组
     *
     * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
     *
     * 字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。
     *
     *
     *
     * 示例 1:
     *
     * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
     * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
     * 示例 2:
     *
     * 输入: strs = [""]
     * 输出: [[""]]
     * 示例 3:
     *
     * 输入: strs = ["a"]
     * 输出: [["a"]]
     *
     */
//    public List<List<String>> groupAnagrams(String[] strs) {
//        Map<String,List<String>>map = new HashMap<>();
//        for (String str : strs) {
//            char[] chars = str.toCharArray();
//            Arrays.sort(chars);
//            String s = new String(chars);
//            List<String> list = map.get(s);
//            if(list==null){
//                List<String> stringList = new ArrayList<>();
//                stringList.add(str);
//                map.put(s,stringList);
//            }else {
//                list.add(str);
//            }
//        }
//        return new ArrayList<>(map.values());
//    }


    /**
     * 旋转图像
     */
    public void rotate(int[][] matrix) {
        int x = matrix.length;
        for (int i = 0; i < x / 2; i++) {
            for (int j = 0; j < (x + 1) / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[x - j - 1][i];
                matrix[x - j - 1][i] = matrix[x - i - 1][x - j - 1];
                matrix[x - i - 1][x - j - 1] = matrix[j][x - i - 1];
                matrix[j][x - i - 1] = temp;
            }
        }
    }

    /**
     * 给你一个 无重复元素 的整数数组candidates 和一个目标整数target，找出candidates中可以使数字和为目标数target
     * 的 所有不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * <p>
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     * <p>
     * 对于给定的输入，保证和为target 的不同组合数少于 150 个。
     * <p>
     * <p>
     * <p>
     * 示例1：
     * <p>
     * 输入：candidates = [2,3,6,7], target = 7
     * 输出：[[2,2,3],[7]]
     * 解释：
     * 2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
     * 7 也是一个候选， 7 = 7 。
     * 仅有这两种组合。
     * 示例2：
     * <p>
     * 输入: candidates = [2,3,5], target = 8
     * 输出: [[2,2,2,2],[2,3,3],[3,5]]
     * 示例 3：
     * <p>
     * 输入: candidates = [2], target = 1
     * 输出: []
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
//        Arrays.sort(candidates);
        backTrack(candidates, 0, target, 0, result, new ArrayList<>());
        return result;
    }

    private void backTrack(int[] candidates, int sum, int target,
                           int index, List<List<Integer>> result,
                           List<Integer> list) {
        if (sum == target) {
            result.add(new ArrayList<>(list));
        }
        if (sum > target) {
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            list.add(candidates[i]);
            backTrack(candidates, sum + candidates[i], target, i, result, list);
            list.remove(list.size() - 1);

        }
    }

    /**
     * 在排序数组中查找元素的第一个和最后一个位置
     * 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
     * <p>
     * 如果数组中不存在目标值 target，返回[-1, -1]。
     * <p>
     * 你必须设计并实现时间复杂度为O(log n)的算法解决此问题。
     */
    public int[] searchRange(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int index = -1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                index = mid;
                break;
            }
        }

        int[] result = new int[2];
        if (index == -1) {
            result[0] = -1;
            result[1] = -1;
        } else {
            int l = index, r = index;
            while (l >= 0 && nums[l] == target) {
                l--;
            }
            while (r <= nums.length - 1 && nums[r] == target) {
                r++;
            }
            result[0] = l + 1;
            result[1] = r - 1;
        }
        return result;
    }

    /**
     * 下一个排列
     */
    public void nextPermutation(int[] nums) {
        int len = nums.length;
        if (len <= 1) return;

        for (int i = len - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {            //找到相邻升序
                for (int j = len - 1; j > i; j--) {
                    if (nums[j] > nums[i]) {    //找到最右边大于nums[i-1]的数，并交换
                        int tmp = nums[i];
                        nums[i] = nums[j];


                        nums[j] = tmp;
                        break;
                    }
                }
                Arrays.sort(nums, i + 1, len);      //将后面降序变为升序
                return;
            }
        }
        Arrays.sort(nums);
    }


    /**
     * 电话号码的字母组合
     * <p>
     * 给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
     * <p>
     * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
     */
    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if (digits.length() == 0) {
            return combinations;
        }
        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backTrack(combinations, phoneMap, digits, 0, new StringBuilder());
        return combinations;
    }

    private void backTrack(List<String> result, Map<Character, String> phoneMap,
                           String digits, int digitIndex, StringBuilder stringBuilder) {
        if (digitIndex == digits.length()) {
            result.add(stringBuilder.toString());
            return;
        }
        String s = phoneMap.get(digits.charAt(digitIndex));
        for (int i = 0; i < s.length(); i++) {
            stringBuilder.append(s.charAt(i));
            backTrack(result, phoneMap, digits, digitIndex + 1, stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
    }


    /**
     * 有效的数独
     * 空白格用 '.' 表示。
     */
    public boolean isValidSudoku(char[][] board) {
        int[][] row = new int[9][9];
        int[][] col = new int[9][9];
        int[][][] san = new int[3][3][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c != '.') {
                    int index = c - '0' - 1;
                    row[i][index]++;
                    col[j][index]++;
                    san[i / 3][j / 3][index]++;
                    if (row[i][index] > 1 || col[j][index] > 1 || san[i / 3][j / 3][index] > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]]
     * 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
     * <p>
     * 你返回所有和为 0 且不重复的三元组。
     * <p>
     * 注意：答案中不可以包含重复的三元组。
     * <p>
     * <p>
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [-1,0,1,2,-1,-4]
     * 输出：[[-1,-1,2],[-1,0,1]]
     * 解释：
     * nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
     * nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
     * nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
     * 不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
     * 注意，输出的顺序和三元组的顺序并不重要。
     * 示例 2：
     * <p>
     * 输入：nums = [0,1,1]
     * 输出：[]
     * 解释：唯一可能的三元组和不为 0 。
     * 示例 3：
     * <p>
     * 输入：nums = [0,0,0]
     * 输出：[[0,0,0]]
     * 解释：唯一可能的三元组和为 0 。
     */
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                if (left != i + 1 && nums[left - 1] == nums[left]) {
                    left++;
                    continue;
                }
                int sum = nums[left] + nums[right] + nums[i];
                if (sum < 0) {
                    left++;
                } else if (sum > 0) {
                    right--;
                } else {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left++]);
                    list.add(nums[right--]);
                    result.add(list);
                }
            }
        }
        return result;
    }


    /**
     * 螺旋矩阵
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        int colLength = matrix.length;
        int rowLength = matrix[0].length;

        int circle = 0;
        List<Integer> list = new ArrayList<>();

        while (true) {
            for (int k = circle; k < rowLength - circle; k++) {
                list.add(matrix[circle][k]);
                if (list.size() == rowLength * colLength) {
                    return list;
                }
            }
            for (int k = circle + 1; k < colLength - circle; k++) {
                list.add(matrix[k][rowLength - circle - 1]);
                if (list.size() == rowLength * colLength) {
                    return list;
                }
            }
            for (int k = rowLength - circle - 2; k >= circle; k--) {
                list.add(matrix[colLength - circle - 1][k]);
                if (list.size() == rowLength * colLength) {
                    return list;
                }
            }
            for (int k = colLength - circle - 2; k > circle; k--) {
                list.add(matrix[k][circle]);
                if (list.size() == rowLength * colLength) {
                    return list;
                }
            }
            circle++;
        }
    }


    /**
     * 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
     * <p>
     * 如果反转后整数超过 32 位的有符号整数的范围[−2的31次方, 2的31次方 − 1] ，就返回 0。
     * <p>
     * 假设环境不允许存储 64 位整数（有符号或无符号）。
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：x = 123
     * 输出：321
     * 示例 2：
     * <p>
     * 输入：x = -123
     * 输出：-321
     * 示例 3：
     * <p>
     * 输入：x = 120
     * 输出：21
     * 示例 4：
     * <p>
     * 输入：x = 0
     * 输出：0
     */
    public int reverse(int x) {
        int result = 0;
        while (x != 0) {
            if (result < Integer.MIN_VALUE / 10 || result > Integer.MAX_VALUE / 10) {
                return 0;
            }
            result = result * 10 + x % 10;
            x = x / 10;
        }

        return result;
    }


    //两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        int jin = 0;
        while (l1 != null || l2 != null) {
            int i1 = l1 == null ? 0 : l1.val;
            int i2 = l2 == null ? 0 : l2.val;
            int sum = jin + i1 + i2;
            jin = sum / 10;
            cur.next = new ListNode(sum % 10);
            cur = cur.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (jin > 0) {
            cur.next = new ListNode(jin);
        }
        return dummy.next;
    }


    /**
     * 给定一个包含非负整数的 mxn网格grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * <p>
     * 说明：一个机器人每次只能向下或者向右移动一步。
     * <p>
     * <p>
     * 输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
     * 输出：7
     * 解释：因为路径 1→3→1→1→1 的总和最小。
     * 示例 2：
     * <p>
     * 输入：grid = [[1,2,3],[4,5,6]]
     * 输出：12
     */
    public int minPathSum(int[][] grid) {
        int x = grid.length;
        int y = grid[0].length;
        int[][] dp = new int[x][y];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < y; i++) {
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        }
        for (int i = 1; i < x; i++) {
            dp[i][0] = grid[i][0] + dp[i - 1][0];
        }
        for (int i = 1; i < x; i++) {
            for (int j = 1; j < y; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[x - 1][y - 1];
    }


    /**
     * 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * <p>
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * <p>
     * 此外，你可以假设该网格的四条边均被水包围。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：grid = [
     * ["1","1","1","1","0"],
     * ["1","1","0","1","0"],
     * ["1","1","0","0","0"],
     * ["0","0","0","0","0"]
     * ]
     * 输出：1
     * 示例 2：
     * <p>
     * 输入：grid = [
     * ["1","1","0","0","0"],
     * ["1","1","0","0","0"],
     * ["0","0","1","0","0"],
     * ["0","0","0","1","1"]
     * ]
     * 输出：3
     */
    public int numIslands(char[][] grid) {
        int x = grid.length;
        int y = grid[0].length;
        int num = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (grid[i][j] == '1') {
                    num++;
                    dfs(grid, i, j);
                }
            }
        }


        return num;
    }

    private void dfs(char[][] grid, int x, int y) {
        int xMax = grid.length;
        int yMax = grid[0].length;
        if (x < 0 || y < 0 || x >= xMax || y >= yMax || grid[x][y] == '0') {
            return;
        }

        grid[x][y] = '0';
        dfs(grid, x - 1, y);
        dfs(grid, x + 1, y);
        dfs(grid, x, y - 1);
        dfs(grid, x, y + 1);
    }

    /**
     * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
     * <p>
     * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     */
    public int findKthLargest(int[] nums, int k) {
        return findKthLargest(nums, 0, nums.length - 1, nums.length - k);
    }

    public int findKthLargest(int[] nums, int left, int right, int index) {
        int baseLeft = left;
        int baseRight = right;

        while (left != right) {
            while (left < right && nums[right] >= nums[baseLeft]) {
                right--;
            }
            while (left < right && nums[left] <= nums[baseLeft]) {
                left++;
            }
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
        }

        int temp = nums[left];
        nums[left] = nums[baseLeft];
        nums[baseLeft] = temp;

        if (left == index) {
            return nums[index];
        }

        if (left < index) {
            return findKthLargest(nums, left + 1, baseRight, index);
        } else {
            return findKthLargest(nums, baseLeft, left - 1, index);
        }
    }


    /**
     * 给你一个整数数组 nums，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     * <p>
     * 测试用例的答案是一个32-位 整数。
     * <p>
     * 子数组 是数组的连续子序列。
     */
    public int maxProduct(int[] nums) {
        int max = Integer.MIN_VALUE, imax = 1, imin = 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(imax * nums[i], nums[i]);
            imin = Math.min(imin * nums[i], nums[i]);
            max = Math.max(max, imax);
        }
        return max;
    }

    /**
     * 416. 分割等和子集
     * 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
     * <p>
     * dp[i][j]  0-i之间是否存在加和等于j
     * <p>
     * nums[i]>j    return false
     * nums[i]==j   dp[i][j] =true
     * num[i]<j     dp[i][j] = dp[i-1][j-num[i]]
     */
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 != 0) {
            return false;//奇数
        }

        int half = sum / 2;
        boolean[] dp = new boolean[half + 1];
        dp[0] = true;
//        if(nums[0]<=half){
//            dp[nums[0]]=true;//初始化首行数据
//        }
        for (int i = 0; i < nums.length; i++) {
            for (int j = half; j >= nums[i]; j--) {
                dp[j] = dp[j] || dp[j - nums[i]];
                if (dp[half]) {
                    return true;
                }
            }
        }
        return false;
    }


    /**
     * 最长连续序列
     * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
     * 请你设计并实现时间复杂度为O(n) 的算法解决此问题。
     * <p>
     * 示例 1：
     * 输入：nums = [100,4,200,1,3,2]
     * 输出：4
     * 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
     * 示例 2：
     * 输入：nums = [0,3,7,2,5,8,4,6,0,1]
     * 输出：9
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }

        int max = 0;
        for (Integer integer : set) {
            if (!set.contains(integer - 1)) {
                int n = 1;
                while (set.contains(integer + 1)) {
                    n++;
                    integer++;
                }
                max = Math.max(n, max);
            }
        }

        return max;
    }


    /**
     * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
     * <p>
     * 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
     * <p>
     * <p>
     * <p>
     * 示例1：
     * <p>
     * 输入：n = 12
     * 输出：3
     * 解释：12 = 4 + 4 + 4
     * 示例 2：
     * <p>
     * 输入：n = 13
     * 输出：2
     * 解释：13 = 4 + 9
     */

    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j++) {

                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }


        }
        return dp[n];
    }


    public int[] merger(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length + arr2.length];
        int step = 0;
        int i = 0;
        int j = 0;
        while (i < arr1.length && j < arr2.length) {
            if (arr1[i] <= arr2[j]) {
                result[step++] = arr1[i++];
            } else {
                result[step++] = arr2[j++];
            }
        }

        if (i == arr1.length) {
            for (int start = j; start < arr2.length; start++) {
                result[step++] = arr2[start];
            }
        } else if (j == arr2.length) {
            for (int start = i; start < arr1.length; start++) {
                result[step++] = arr1[start];
            }
        }
        return result;
    }

    //2023.3.3回顾

    //全排列
    public List<String> rank(String s) {
        List<String> result = new ArrayList<>();
        char[] chars = s.toCharArray();
        rank(result, 0, chars);
        return result;
    }

    private void rank(List<String> result, int start, char[] chars) {
        if (start == chars.length) {
            result.add(new String(chars));
            return;
        }

        for (int i = start; i < chars.length; i++) {
            swap(chars, start, i);
            rank(result, start + 1, chars);
            swap(chars, start, i);
        }


    }

    private void swap(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    //合法的括号匹配
    public List<String> getBrackets(int n) {
        List<String> result = new ArrayList<>();
        getBrackets(n, 0, 0, result, new StringBuilder());
        return result;
    }

    public void getBrackets(int n, int leftCount, int rightCount, List<String> result, StringBuilder stringBuilder) {
        if (leftCount + rightCount == 2 * n) {
            result.add(stringBuilder.toString());
            return;
        }

        if (leftCount < n) {
            stringBuilder.append("(");
            getBrackets(n, leftCount + 1, rightCount, result, stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }

        if (rightCount < leftCount) {
            stringBuilder.append(")");
            getBrackets(n, leftCount, rightCount + 1, result, stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
    }


    //反转括号内字符串
    private String reverse(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        Stack<String> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(stringBuilder.toString());
                stringBuilder.setLength(0);
            } else if (c == ')') {
                stringBuilder.reverse();
                stringBuilder.insert(0, stack.pop());
            } else {
                stringBuilder.append(c);
            }
        }
        return stringBuilder.toString();
    }


    //找出最长有效（格式正确且连续）括号子串的长度
    private int getLength(String s) {
        int max = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    max = Math.max(max, i - stack.peek());
                }
            }
        }
        return max;
    }


    //最长回文子串
    private String getMaxRepeat(String s) {
        int maxIndex = 0;
        int maxLength = 0;
        for (int i = 0; i < s.length(); i++) {
            int maxLength1 = getMaxRepeat(i, i, s);
            int maxLength2 = getMaxRepeat(i, i + 1, s);
            int maxLength3 = Math.max(maxLength1, maxLength2);
            if (maxLength3 >= maxLength) {
                maxIndex = i;
                maxLength = maxLength3;
            }
        }
        return s.substring(maxIndex - (maxLength - 1) / 2, maxIndex + maxLength / 2 + 1);
    }


    private int getMaxRepeat(int leftIndex, int rightIndex, String s) {
        while (leftIndex >= 0 && rightIndex < s.length()) {
            if (s.charAt(leftIndex) != s.charAt(rightIndex)) {
                break;
            }
            leftIndex--;
            rightIndex++;
        }
        return rightIndex - leftIndex - 1;
    }

    /**
     * 给定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。
     * <p>
     * 示例 1 :
     * <p>
     * 输入: 2736
     * 输出: 7236
     * 解释: 交换数字2和数字7。
     * 示例 2 :
     * <p>
     * 输入: 9973
     * 输出: 9973
     * 解释: 不需要交换。
     */

    public int maximumSwap(int num) {
        String s = String.valueOf(num);
        int maxLeft = 0;
        int maxRight = 0;
        for (int i = 0; i < s.length(); i++) {
            int maxChar = s.charAt(i) - '0';
            maxLeft = i;
            maxRight = i;
            for (int j = i + 1; j < s.length(); j++) {
                int rightChar = s.charAt(j) - '0';
                if (rightChar >= maxChar) {
                    maxChar = rightChar;
                    maxRight = j;
                }
            }
            if (s.charAt(i) - '0' != s.charAt(maxRight) - '0') {//如果左边不是最大值,直接交换就好
                break;
            }
        }
        char[] chars = s.toCharArray();
        swap(chars, maxLeft, maxRight);
        String result = new String(chars);
        return Integer.parseInt(result);
    }

    //单调递增子序列长度
    public int getLength(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int max = 1;
        for (int i = 1; i < nums.length; i++) {
            int dpMax = 1;
            for (int j = 0; j < dp.length; j++) {
                if (nums[i] > nums[j]) {
                    dpMax = Math.max(dpMax, dp[j] + 1);
                }
            }

            dp[i] = dpMax;
            max = Math.max(dpMax, max);
        }
        return max;
    }


    /**
     * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
     * <p>
     * 回文串 是正着读和反着读都一样的字符串。
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：s = "aab"
     * 输出：[["a","a","b"],["aa","b"]]
     * 示例 2：
     * <p>
     * 输入：s = "a"
     * 输出：[["a"]]
     */
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        List<String> list = new ArrayList<>();

        int[][] arrs = new int[s.length()][s.length()];
        for (int i = 0; i < arrs.length; i++) {
            Arrays.fill(arrs[i], 0);
        }
        dfs(s, 0, list, result, arrs);
        return result;
    }


    private void dfs(String s, int start, List<String> list, List<List<String>> result, int[][] arrs) {
        if (start == s.length()) {
            result.add(new ArrayList<>(list));
        }


        for (int i = start; i < s.length(); i++) {
            if (isPalindrome(start, i, arrs, s) == 1) {
                list.add(s.substring(start, i + 1));
                dfs(s, i + 1, list, result, arrs);
                list.remove(list.size() - 1);
            }
        }
    }

    private int isPalindrome(int i, int j, int[][] arrs, String s) {
        if (arrs[i][j] != 0) {
            return arrs[i][j];
        }

        if (i >= j) {
            arrs[i][j] = 1;
        } else if (s.charAt(i) == s.charAt(j)) {
            arrs[i][j] = isPalindrome(i + 1, j - 1, arrs, s);
        } else {
            arrs[i][j] = -1;
        }

        return arrs[i][j];
    }


    /**
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     * <p>
     * 如果不存在公共前缀，返回空字符串 ""。
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }

        int length = strs[0].length();
        for (int i = 0; i < length; i++) {
            char c = strs[0].charAt(i);
            boolean isSame = true;
            for (int j = 0; j < strs.length; j++) {
                if (strs[j].length() == i || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }

        }
        return strs[0];
    }


    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。
     * <p>
     * <p>
     * 输入: s = "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     * 示例 2:
     * <p>
     * 输入: s = "bbbbb"
     * 输出: 1
     * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
     * 示例 3:
     * <p>
     * 输入: s = "pwwkew"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是"wke"，所以其长度为 3。
     * 请注意，你的答案必须是 子串 的长度，"pwke"是一个子序列，不是子串。
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int preMax = 1;
        int max = 1;
        for (int i = 1; i < s.length(); i++) {
            String temp = s.substring(i - preMax, i);//前一个元素不重复的最长字符串
            char c = s.charAt(i);
            int index = temp.indexOf(c);//目前字符在前一个元素不重复的最长字符串d的最后索引
            if (index != -1) {
                preMax = temp.length() - index;
            } else {
                preMax = preMax + 1;
            }
            max = Math.max(max, preMax);
        }
        return max;
    }

    /**
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * <p>
     * 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [10,9,2,5,3,7,101,18]
     * 输出：4
     * 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
     * 示例 2：
     * <p>
     * 输入：nums = [0,1,0,3,2,3]
     * 输出：4
     * <p>
     * <p>
     * 示例 3：
     * <p>
     * 输入：nums = [7,7,7,7,7,7,7]
     * 输出：1
     */


    public int lengthOfLIS(int[] nums) {
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > list.get(list.size() - 1)) {
                list.add(nums[i]);
            } else {
                int left = 0, right = list.size() - 1, mid;
                while (left <= right) {
                    mid = (left + right) / 2;
                    if (list.get(mid) < nums[i]) {
                        left = mid + 1;
                    } else if (list.get(mid) > nums[i]) {
                        right = mid - 1;
                    } else {
                        left = mid;
                        break;
                    }
                }
                list.set(left, nums[i]);
            }
        }

        return list.size();
    }


    /**
     * 盛最多水的容器
     */
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while (left < right) {
            int x = right - left;
            int y = Math.min(height[left], height[right]);
            max = Math.max(max, x * y);
            //往左找一个比height[left]大的
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return max;
    }


    /**
     * 三数之和
     */
    public List<List<Integer>> threeSum1(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {//元素重复跳过
                continue;
            }
            //判断两数之和
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                if (left != i + 1 && nums[left] == nums[left - 1]) {
                    left++;
                    continue;
                }
                if (nums[left] + nums[right] + nums[i] < 0) {
                    left++;
                } else if (nums[left] + nums[right] + nums[i] > 0) {
                    right--;
                } else {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[left]);
                    list.add(nums[right]);
                    list.add(nums[i]);
                    res.add(list);
                    left++;
                    right--;
                }
            }
        }
        return res;
    }

    /**
     * 移动和
     */
    public void moveZeroes(int[] nums) {
        int left = 0;
        int right = 0;
        while (right < nums.length) {
            if (nums[right] != 0) {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
            }
            right++;
        }
    }

    /**
     * 给你一个只包含 '('和 ')'的字符串，找出最长有效（格式正确且连续）括号子串的长度。
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：s = "(()"
     * 输出：2
     * 解释：最长有效括号子串是 "()"
     * 示例 2：
     * <p>
     * 输入：s = ")()())"
     * 输出：4
     * 解释：最长有效括号子串是 "()()"
     * 示例 3：
     * <p>
     * 输入：s = ""
     * 输出：0
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }

        int max = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    max = Math.max(max, i - stack.peek());
                }
            }
        }

        return max;
    }

    /**
     * 给你一个整数数组 nums，请你找出并返回能被三整除的元素最大和
     */
    public int maxSumDivThree(int[] nums) {
        int[] remainder = new int[3];
        for (int num : nums) {
            int a = remainder[0] + num;
            int b = remainder[1] + num;
            int c = remainder[2] + num;

            remainder[a % 3] = Math.max(remainder[a % 3], a);
            remainder[b % 3] = Math.max(remainder[b % 3], b);
            remainder[c % 3] = Math.max(remainder[c % 3], c);
        }
        return remainder[0];


    }


    /**
     * 接雨水
     * <p>
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水
     */

    public int trap(int[] height) {
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        int leftMax = 0;
        int rightMax = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] > leftMax) {
                    leftMax = height[left];
                } else {
                    result += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] > rightMax) {
                    rightMax = height[right];
                } else {
                    result += rightMax - height[right];
                }
                right--;
            }
        }
        return result;
    }

    /**
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * <p>
     * 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：s = "babad"
     * 输出："bab"
     * 解释："aba" 同样是符合题意的答案。
     * 示例 2：
     * <p>
     * 输入：s = "cbbd"
     * 输出："bb"
     */

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 2) {
            return "";
        }

        String result = "";
        for (int i = 0; i < s.length(); i++) {
            String oneString = getPalindromeLen(s, i, i);
            String twoString = getPalindromeLen(s, i, i + 1);
            String tempMax = oneString.length() > twoString.length() ? oneString : twoString;
            result = tempMax.length() > result.length() ? tempMax : result;
        }
        return result;
    }

    private String getPalindromeLen(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left + 1, right);
    }


    /**
     * 幂集。编写一种方法，返回某集合的所有子集。集合中不包含重复的元素。
     * 输入： nums = [1,2,3]
     * 输出：
     * [3],
     * [1],
     * [2],
     * [1,2,3],
     * [1,3],
     * [2,3],
     * [1,2],
     * []
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        subsets(nums, 0, list, result);
        return result;
    }

    public void subsets(int[] nums, int index, List<Integer> list, List<List<Integer>> result) {
        if (index == nums.length) {
            result.add(new ArrayList<>(list));
            return;
        }

        //选择index位置上的数
        list.add(nums[index]);
        subsets(nums, index + 1, list, result);

        //不选择index位置上的数
        list.remove(list.size() - 1);
        subsets(nums, index + 1, list, result);

    }


    static class NQueens {

        public List<Map<Integer, Integer>> getNQueens(int n) {
            List<Map<Integer, Integer>> list = new ArrayList<>();
            int[] segment = new int[n];
            fill(0, n, segment, list);
            return list;
        }

        private void fill(int row, int n, int[] segment, List<Map<Integer, Integer>> list) {
            if (row == n) {
                Map<Integer, Integer> queens = new TreeMap<>();
                for (int i = 0; i < segment.length; i++) {
                    queens.put(i, segment[i]);
                }
                list.add(queens);
                return;
            }

            int index = 0;
            int[] unused = new int[n - row];
            for (int i = 0; i < n; i++) {
                boolean flag = true;
                for (int j = 0; j < row; j++) {
                    if (i == segment[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    unused[index++] = i;
                }
            }

            for (int i : unused) {
                segment[row] = i;
                fill(row + 1, n, segment, list);
            }
        }

        @SuppressWarnings("SameParameterValue")
        private void print(int n) {
            List<Map<Integer, Integer>> nQueens = getNQueens(n);
            for (Map<Integer, Integer> map : nQueens) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (j == map.get(i)) {
                            System.out.print("Q");
                        } else {
                            System.out.print(" ");
                        }
                        if (j == n - 1) {
                            System.out.println();
                        }
                    }
                }
                System.out.println("--------------------");
            }
        }
    }

    static class Shape {
        public void diamond(int sideLength, boolean isCross) {
            for (int row = 0; row < 2 * sideLength - 1; row++) {
                for (int col = 0; col < 2 * sideLength - 1; col++) {
                    if (row < sideLength) {
                        if (col == sideLength - row - 1 ||
                                col == sideLength + row - 1 ||
                                (isCross && row > sideLength / 2 &&
                                        (col == row || col == 2 * sideLength - row - 2))) {
                            System.out.print("*");
                        } else {
                            System.out.print(" ");
                        }
                    } else {
                        if (col == row - sideLength + 1 ||
                                col == 3 * sideLength - row - 3 ||
                                (isCross && row < 1.5 * (sideLength - 1) &&
                                        (col == 2 * sideLength - row - 2 || col == row))) {
                            System.out.print("*");
                        } else {
                            System.out.print(" ");
                        }
                    }
                }
                System.out.println();
            }
        }
    }

    static class ListUtil {
        static class ListNode {
            int val;
            ListNode next = null;

            ListNode(int val) {
                this.val = val;
            }

            @Override
            public String toString() {
                return "ListNode{" +
                        "val=" + val +
                        ", next=" + next +
                        '}';
            }
        }

        /**
         * 相交链表
         * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
         */
        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            ListNode node = null;
            ListNode an = headA;
            ListNode bn = headB;
            while (an != bn) {
                an = an == null ? headB : an.next;
                bn = bn == null ? headA : bn.next;
            }
            return an;
        }


        //两两交换链表中的节点
        public ListNode swapPairs(ListNode head) {
            ListNode dummy = new ListNode(-1);
            dummy.next = head;
            ListNode cur = dummy;
            while (cur.next != null && cur.next.next != null) {

                ListNode node1 = cur.next;
                ListNode node2 = cur.next.next;

                cur.next = node2;
                node1.next = node2.next;
                node2.next = node1;

                cur = node1;
            }
            return dummy.next;
        }


        public boolean isEqual(ListNode listNode1, ListNode listNode2) {
            while (listNode1 != null && listNode2 != null) {
                if (listNode1.val != listNode2.val) {
                    return false;
                }
                listNode1 = listNode1.next;
                listNode2 = listNode2.next;
            }
            return listNode1 == null && listNode2 == null;
        }


        //删除倒数第n个节点
        public ListNode removeNthFromEnd(ListNode head, int n) {
            ListNode fast = head;
            ListNode dummy = new ListNode(-1);
            dummy.next = head;
            ListNode slow = dummy;
            for (int i = 0; i < n; i++) {
                fast = fast.next;
            }

            while (fast != null) {
                fast = fast.next;
                slow = slow.next;
            }

            if (slow.next != null) {
                slow.next = slow.next.next;
            }
            return dummy.next;
        }

        public static ListNode reverseList(ListNode head) {
            ListNode cur = head;
            ListNode pre = null;
            while (cur != null) {
                ListNode next = cur.next;
                cur.next = pre;
                pre = cur;
                cur = next;
            }
            return pre;
        }

        public static boolean hasCycle(ListNode head) {
            ListNode fast = head;


            while (fast != null && fast.next != null) {
                fast = fast.next.next;
                head = head.next;
                if (fast == head) {
                    return true;
                }
            }
            return false;
        }

        //将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
        public static ListNode mergeList(ListNode list1, ListNode list2) {
            ListNode listNode = new ListNode(-1);
            ListNode head = listNode;
            while (list1 != null && list2 != null) {
                if (list1.val >= list2.val) {
                    head.next = list2;
                    list2 = list2.next;
                } else {
                    head.next = list1;
                    list1 = list1.next;
                }
                head = head.next;
            }

            if (list1 == null) {
                head.next = list2;
            } else {
                head.next = list1;
            }
            return listNode.next;
        }
    }

    static class Permutation {
        /**
         * @author lilei
         * create at 2020/8/15 17:20
         * 给定一个字符序列，返回所有不同序列的情况
         */
        public <E> Set<List<E>> rank(E[] es) {
            Set<List<E>> all = new HashSet<>();
            boolean[] used = new boolean[es.length];
            int[] signal = new int[es.length];
            dfs(0, all, signal, es, used);
            return all;
        }

        private <E> void dfs(int deep, Set<List<E>> all,
                             int[] signal, E[] es, boolean[] used) {
            if (deep == es.length) {
                List<E> list = new ArrayList<>();
                for (int i = 0; i < es.length; i++) {
                    list.add(es[signal[i]]);
                }
                all.add(list);
                return;
            }

            for (int i = 0; i < es.length; i++) {
                if (used[i]) {
                    continue;
                }
                signal[deep] = i;
                used[i] = true;
                dfs(deep + 1, all, signal, es, used);
                used[i] = false;
            }
        }


        public List<List<Integer>> permute(Integer[] nums) {
            List<Integer> ints = Arrays.asList(nums);
            List<List<Integer>> result = new ArrayList<>();
            backtrack(nums.length, ints, result, 0);
            return result;
        }

        private void backtrack(int length, List<Integer> list, List<List<Integer>> result, int index) {
            if (index == length) {
                result.add(new ArrayList<>(list));
                return;
            }

            for (int i = index; i < length; i++) {
                Collections.swap(list, index, i);
                backtrack(length, list, result, index + 1);
                Collections.swap(list, index, i);
            }
        }
    }

    static class Ip {
        private final List<String> ips = new ArrayList<>();
        private String ip;
        private final int[] segment = new int[3];

        /**
         * @author lilei
         * create at 13:55 2020/8/7
         * 给定ip字符串返回所有可能的ip地址
         * 比如给定25525512122，可能的ip为255.255.12.122或者255.255.121.22
         **/
        public List<String> getIps(String ip) {
            if (isNotNumber(ip)) return ips;
            this.ip = ip;
            addIp(0, 0, 0);
            return ips;
        }


        /**
         * @param count:第几个点
         * @param index:点的下标
         **/
        private void addIp(int count, int index, int height) {

            int last = ip.length() - index;

            if (last > (4 - count) * 3 || last < (4 - count)) {
                return;
            }

            char nextChar = ip.charAt(index);

            //last segment should not a** (a>2)
            if (count == 3
                    && getLastSegment(ip, index) <= 255
                    && !(nextChar == '0' && last > 1)) {
                ips.add(split());
                return;
            }

            int max = nextChar == '0' ? 1 : nextChar > '2' ? 2 : 3;

            for (int i = 0; i < max && index + i + 1 < ip.length(); i++) {

                String substring = ip.substring(index, index + i + 1);

                if (getLastSegment(substring, 0) > 255) {
                    continue;
                }


                if (height < 3)
                    segment[height] = index + i + 1;

                addIp(count + 1, index + i + 1, height + 1);
            }
        }

        private boolean isNotNumber(String s) {
            char[] chars = s.toCharArray();
            for (char c : chars) {
                if (c == '.') continue;
                if (c <= '0' || c >= '9') {
                    return true;
                }
            }
            return false;
        }

        private int getLastSegment(String ip, int index) {
            int re = 0;
            for (; index < ip.length(); index++) {
                re = re * 10 + ip.charAt(index) - '0';
            }
            return re;
        }

        private String split() {
            StringBuilder stringBuilder = new StringBuilder(ip);
            stringBuilder.insert(segment[0], ".");
            stringBuilder.insert(segment[1] + 1, ".");
            stringBuilder.insert(segment[2] + 2, ".");
            return stringBuilder.toString();
        }

        public boolean validIp4(String ip) {
            String[] split = ip.split("\\.");
            for (String s : split) {
                if (s == null || s.length() == 0) return false;
                if (s.length() > 3) return false;
                if (s.length() != 1 && s.charAt(0) == '0') return false;
                if (getLastSegment(s, 0) > 255) return false;
                if (isNotNumber(s)) return false;
            }
            return true;
        }
    }

    static class MedianFinder {
        private final PriorityQueue<Double> maxheap = new PriorityQueue<>((x, y) -> (int) (y - x));
        private final PriorityQueue<Double> minheap = new PriorityQueue<>();

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
        }

        public void addNum(double num) {
            if (minheap.isEmpty() || minheap.peek() <= num) {
                minheap.offer(num);
            } else {
                maxheap.offer(num);
            }
            if (minheap.size() - maxheap.size() == 2) {
                maxheap.offer(minheap.poll());
            }
            if (maxheap.size() - minheap.size() == 2) {
                minheap.offer(maxheap.poll());
            }
        }

        public double findMedian(List<Double> list) {
            for (Double d : list) {
                addNum(d);
            }
            return findMedian();
        }


        public double findMedian() {
            Double max = maxheap.peek();
            Double min = minheap.peek();
            max = max == null ? 0 : max;
            min = min == null ? 0 : min;
            if (maxheap.size() == minheap.size()) {
                return (max + min) / 2;
            }
            if (maxheap.size() > minheap.size()) {
                return max;
            }
            return min;
        }

    }

    static class Counter {
        /**
         * @author lilei
         * create at 11:54 2020/8/11
         * 前K个出现频率最高的元素
         **/
        public <E> List<E> topK(E[] element, int k) {

            Map<E, Integer> count = new HashMap<>();

            for (E e : element) {
                count.merge(e, 1, Integer::sum);
            }

            PriorityQueue<E> maxheap =
                    new PriorityQueue<>(((o1, o2) -> count.get(o2) - count.get(o1)));

            for (E e : count.keySet()) {
                maxheap.offer(e);
            }

            List<E> es = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                E poll = maxheap.poll();
                if (poll == null) {
                    break;
                }
                es.add(poll);
            }

            return es;
        }


        /**
         * @author lilei
         * create at 10:48 2020/7/28
         * 求给定数组连续子数组的最大值（dp求解）
         * 时间复杂度O(n) 空间复杂度O(1)
         **/
        public int maxSubArray(int[] nums) {
            int res = nums[0];
            int max = 0;
            for (int i = 1; i < nums.length; i++) {
                max = Math.max(max + nums[i], 0);
                res = Math.max(max, res);
            }
            return res;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 最大连续e的个数,e如果为null，返回此数组最大连续元素的个数
         **/
        public <E> int maxRepeat(E[] es, E e) {
            int max = 0;
            int p = 0;
            while (p < es.length) {
                if (es[p] != e && e != null) {
                    p++;
                    continue;
                }
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                max = Math.max(step, max);
                p = pn;
            }
            return max;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es最大连续相同元素
         **/
        public <E> List<E> maxRepeat(E[] es) {
            int max = 0;
            int p = 0;
            List<E> maxEs = new ArrayList<>();
            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                if (max == step) {
                    maxEs.add(es[p]);
                } else {
                    max = Math.max(step, max);
                    if (max == step) {
                        maxEs.clear();
                        maxEs.add(es[p]);
                    }
                }
                p = pn;
            }
            return maxEs;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es最大连续相同元素与元素的个数
         **/
        public <E> Map.Entry<List<E>, Integer> maxRepeatAndCount(E[] es) {
            int max = 0;
            int p = 0;
            List<E> maxEs = new ArrayList<>();
            Map.Entry<List<E>, Integer> maxEsIn = new HashMap.SimpleEntry<>(maxEs, 0);

            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                if (max == step) {
                    maxEs.add(es[p]);
                } else {
                    max = Math.max(step, max);
                    if (max == step) {
                        maxEs.clear();
                        maxEs.add(es[p]);
                    }
                }
                p = pn;
            }
            maxEsIn.setValue(max);
            return maxEsIn;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es连续相同元素与元素的个数(同元素取最大个数)
         **/
        public <E> Map<Integer, List<E>> repeatAndCount(E[] es) {
            int p = 0;
            Map<E, Integer> countMap = new HashMap<>();

            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                Integer count = countMap.get(es[p]);
                if (count == null) {
                    countMap.put(es[p], step);
                } else {
                    if (step > count) {
                        countMap.put(es[p], step);
                    }
                }
                p = pn;
            }
            return mergeSameValue(countMap);
        }

        private <E> List<E> removeRepeat(E[] es) {
            List<E> list = new ArrayList<>();
            int p = 0;
            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                list.add(es[p]);
                p = pn;
            }
            return list;
        }

        private String removeDuplicate(String s) {
            Stack<Character> stack = new Stack<>();
            HashSet<Character> seen = new HashSet<>();
            HashMap<Character, Integer> last_occurrence = new HashMap<>();
            for (int i = 0; i < s.length(); i++) last_occurrence.put(s.charAt(i), i);
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (!seen.contains(c)) {
                    while (!stack.isEmpty() && c < stack.peek() && last_occurrence.get(stack.peek()) > i) {
                        seen.remove(stack.pop());
                    }
                    seen.add(c);
                    stack.push(c);
                }
            }
            StringBuilder sb = new StringBuilder(stack.size());
            for (Character c : stack) sb.append(c.charValue());
            return sb.toString();
        }

        private <E> Map<Integer, List<E>> mergeSameValue(Map<E, Integer> map) {
            Map<Integer, List<E>> reMap = new HashMap<>();
            for (Map.Entry<E, Integer> e : map.entrySet()) {
                Integer value = e.getValue();
                List<E> es = reMap.get(value);
                if (es == null) {
                    es = new ArrayList<>();
                }
                es.add(e.getKey());
                reMap.put(e.getValue(), es);
            }
            return reMap;
        }
    }

    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        partition(res, list, n, 0, new HashSet<>(), new HashSet<>(), new HashSet<>());
        return res;
    }

    public void partition(List<List<String>> res, List<Integer> list, int n, int row, Set<Integer> leftSet, Set<Integer> rightSet, Set<Integer> set) {
        if (list.size() == n) {
            List<String> stringList = new ArrayList<>();
            for (Integer integer : list) {
                char[] chars = new char[n];
                for (int i = 0; i < n; i++) {
                    if (i == integer) {
                        chars[i] = 'Q';
                    } else {
                        chars[i] = '.';
                    }
                }
                stringList.add(new String(chars));
            }
            res.add(stringList);
            return;
        }

        for (int i = 0; i < n; i++) {
            if (set.contains(i)) {
                continue;
            }
            if (rightSet.contains(i + row)) {
                continue;
            }
            if (leftSet.contains(i - row)) {
                continue;
            }

            set.add(i);
            leftSet.add(i - row);
            rightSet.add(i + row);
            list.add(i);
            partition(res, list, n, row + 1, leftSet, rightSet, set);
            list.remove(list.size() - 1);
            set.remove(i);
            leftSet.remove(i - row);
            rightSet.remove(i + row);
        }
    }

    //79. 单词搜索
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (exist(board, word, 0, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean exist(char[][] board, String word, int step, int i, int j) {
        int index = word.indexOf(step);
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(step)) {
            return false;
        }

        if (step == word.length() - 1) {
            return true;
        }

        char temp = board[i][j];
        board[i][j] = ' ';
        if (
                exist(board, word, step + 1, i + 1, j) ||
                        exist(board, word, step + 1, i, j + 1) ||
                        exist(board, word, step + 1, i - 1, j) ||
                        exist(board, word, step + 1, i, j - 1)) {
            return true;
        }
        board[i][j] = temp;

        return false;
    }

    //分格回文串
    public List<List<String>> partition1(String s) {
        List<List<String>> res = new ArrayList<>();
        List<String> list1 = new ArrayList<>();
        int[][] dp = new int[s.length()][s.length()];
        partition(s, 0, res, list1, dp);
        return res;
    }

    public void partition(String s, int start, List<List<String>> res, List<String> list, int[][] dp) {
        if (start >= s.length()) {
            res.add(new ArrayList<>(list));
        }

        for (int i = start; i < s.length(); i++) {
            if (isHuiWen(s, start, i, dp) == 1) {
                list.add(s.substring(start, i + 1));
                partition(s, i + 1, res, list, dp);
                list.remove(list.size() - 1);
            }
        }
    }

    public int isHuiWen(String s, int i, int j, int[][] dp) {
        if (dp[i][j] != 0) {
            return dp[i][j];
        }

        if (i >= j) {
            return 1;
        }

        if (s.charAt(i) == s.charAt(j)) {
            return isHuiWen(s, i + 1, j - 1, dp);
        } else {
            dp[i][j] = -1;
            return -1;
        }

    }

    /**
     * 请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。
     * <p>
     * 注意，您的结果中 不应 包含任何括号。
     * 示例 1：
     * <p>
     * 输入：s = “(abcd)”
     * 输出：“dcba”
     * <p>
     * 示例 2：
     * <p>
     * 输入：s = “(u(love)i)”
     * 输出：“iloveu”
     * <p>
     * 示例 3：
     * <p>
     * 输入：s = “(ed(et(oc))el)”
     * 输出：“leetcode”
     */
    public String reverseBracketStr(String s) {
        LinkedList<String> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                stack.push(sb.toString());
                sb.setLength(0);
            } else if (ch == ')') {
                sb.reverse();
                sb.insert(0, stack.pop());
            } else {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        backtrack(ans, new StringBuilder(), 0, 0, n);
        return ans;
    }

    public void backtrack(List<String> ans, StringBuilder cur, int open, int close, int max) {
        if (cur.length() == max * 2) {
            ans.add(cur.toString());
            return;
        }
        if (open < max) {
            cur.append('(');
            backtrack(ans, cur, open + 1, close, max);
            cur.deleteCharAt(cur.length() - 1);
        }
        if (close < open) {
            cur.append(')');
            backtrack(ans, cur, open, close + 1, max);
            cur.deleteCharAt(cur.length() - 1);
        }
    }


    //爬楼梯
    //假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    //每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int[] ints = new int[n + 1];
        ints[2] = 2;
        ints[1] = 1;
        for (int i = 3; i <= n; i++) {
            ints[i] = ints[i - 1] + ints[i - 2];
        }
        return ints[n];
    }

    //杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> list1 = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    list1.add(1);
                } else {
                    list1.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
                }
            }
            res.add(list1);
        }
        return res;
    }

    //买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int min = prices[0];
        int profit = 0;
        for (int price : prices) {
            if (price < min) {
                min = price;
            } else {
                profit = Math.max(price - min, profit);
            }
        }
        return profit;
    }

    //跳跃游戏
    public boolean canJump(int[] nums) {
        int needJump = 1;//最少需要跳的步数
        for (int i = nums.length - 2; i >= 0; i--) {
            if (nums[i] < needJump) {
                needJump++;
            } else {
                needJump = 1;
            }
        }
        return needJump == 1;
    }

    //跳跃游戏II
    //给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
    //
    //每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
    //
    //0 <= j <= nums[i]
    //i + j < n
    //返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
    public int jumpDP(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 0;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = nums.length;
            for (int j = 0; j < i; j++) {
                if (j + nums[j] >= i) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        if (dp[nums.length - 1] >= nums.length) {
            return 0;
        }

        return dp[nums.length - 1];
    }

    //贪心 每一步都选择更远的位置
    public int jumpTX(int[] nums) {
        int step = 0;
        int maxIndex = 0;
        int end = 0;
        for (int i = 0; i < nums.length; i++) {
            maxIndex = Math.max(maxIndex, i + nums[i]);//每次step最大的索引

            if (i == end) {//如果走到了最大的索引那就代表
                end = maxIndex;
                step++;
            }
        }

        return step;
    }

    //划分字母区间
    //给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。
    //
    //注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。
    //
    //返回一个表示每个字符串片段的长度的列表。
    public List<Integer> partitionLabels(String s) {
        List<Integer> list1 = new ArrayList<>();
        int maxIndex = 0;
        int beginIndex = -1;

        for (int i = 0; i < s.length(); i++) {
            int last = s.lastIndexOf(s.charAt(i));
            maxIndex = Math.max(last, maxIndex);

            if (i == maxIndex) {
                list1.add(maxIndex - beginIndex);
                beginIndex = maxIndex;
            }
        }

        return list1;
    }

    //腐烂的橘子
    //值 0 代表空单元格；
    //值 1 代表新鲜橘子；
    //值 2 代表腐烂的橘子。
    public int orangesRotting(int[][] grid) {
        int day = -1;
        Queue<List<Pair<Integer, Integer>>> queue = new LinkedList<>();//新腐烂的橘子
        int rotNums = 0;
        int totalNums = 0;
        List<Pair<Integer, Integer>> list1 = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {

                if (grid[i][j] == 2) {
                    Pair<Integer, Integer> pair = new Pair<>(i, j);
                    list1.add(pair);
                }

                if (grid[i][j] == 2 || grid[i][j] == 1) {
                    totalNums++;
                }
            }
        }
        queue.offer(list1);

        boolean hasRot = true;
        while (hasRot && !queue.isEmpty()) {
            day++;
            List<Pair<Integer, Integer>> poll = queue.poll();

            List<Pair<Integer, Integer>> list2 = new ArrayList<>();
            for (Pair<Integer, Integer> pair : poll) {
                int x = pair.getKey();
                int y = pair.getValue();

                rotNums++;

                if (x + 1 < grid.length && grid[x + 1][y] == 1) {//新鲜的橘子被感染
                    grid[x + 1][y] = 2;
                    list2.add(new Pair<>(x + 1, y));
                }

                if (y + 1 < grid[0].length && grid[x][y + 1] == 1) {//新鲜的橘子被感染
                    grid[x][y + 1] = 2;
                    list2.add(new Pair<>(x, y + 1));
                }

                if (x - 1 >= 0 && grid[x - 1][y] == 1) {//新鲜的橘子被感染
                    grid[x - 1][y] = 2;
                    list2.add(new Pair<>(x - 1, y));
                }

                if (y - 1 >= 0 && grid[x][y - 1] == 1) {//新鲜的橘子被感染
                    grid[x][y - 1] = 2;
                    list2.add(new Pair<>(x, y - 1));
                }
            }

            if (!list2.isEmpty()) {
                queue.add(list2);
            } else {
                hasRot = false;
            }
        }

        if (rotNums == totalNums) {
            return day;
        }

        return -1;
    }
    //课程表
    //你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
    //
    //在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
    //
    //例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
    //请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] in = new int[numCourses];
        List<List<Integer>> out = new ArrayList<>();

        for (int i = 0; i < numCourses; i++) {
            out.add(new ArrayList<>());

        }


        //计算每个节点的出度集合
        for (int[] prerequisite : prerequisites) {
            List<Integer> integers = out.get(prerequisite[0]);
            integers.add(prerequisite[1]);
            in[prerequisite[1]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        //将入度为0的入队
        for (int i = 0; i < numCourses; i++) {
            if (in[i] == 0) {
                queue.add(i);
            }
        }

        int step = 0;
        while (!queue.isEmpty()) {
            step++;
            Integer poll = queue.poll();
            List<Integer> integers = out.get(poll);

            if (integers != null) {
                for (Integer integer : integers) {
                    in[integer]--;
                    if (in[integer] == 0) {
                        queue.add(integer);
                    }
                }
            }
        }

        return step == numCourses;
    }

    //搜索插入位置
    //给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
    //
    //请必须使用时间复杂度为 O(log n) 的算法。
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                return mid;
            }
        }
        return left + 1;

    }

    //搜索二维矩阵
    //给你一个满足下述两条属性的 m x n 整数矩阵：
    //
    //每行中的整数从左到右按非严格递增顺序排列。
    //每行的第一个整数大于前一行的最后一个整数。
    //给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
    public boolean searchMatrix1(int[][] matrix, int target) {
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][matrix[0].length - 1] >= target) {
                int left = 0;
                int right = matrix[0].length - 1;
                while (left <= right) {
                    int mid = (left + right) / 2;
                    if (matrix[i][mid] < target) {
                        left = mid + 1;
                    } else if (matrix[i][mid] > target) {
                        right = mid - 1;
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;
    }


    /**
     * 搜索二维矩阵 II
     * 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
     * <p>
     * 每行的元素从左到右升序排列。
     * 每列的元素从上到下升序排列。
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int i=0;
        int j=matrix[0].length-1;
        while (i<matrix.length&&j>=0){
            if(matrix[i][j]>target){
                j--;
            }else if(matrix[i][j]<target){
                i++;
            }else {
                return true;
            }
        }
        return false;
    }

    //在排序数组中查找元素的第一个和最后一个位置
    //给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
    //
    //如果数组中不存在目标值 target，返回 [-1, -1]。
    //
    //你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
    public int[] searchRange1(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int index = -1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                index = mid;
                break;
            }
        }

        if (index == -1) {
            return new int[]{-1, -1};
        }

        int l = index;
        int r = index;
        while (l >= 0 && nums[l] == target) {
            l--;
        }
        while (r < nums.length && nums[r] == target) {
            r++;
        }

        int[] ints = new int[2];
        ints[0] = l + 1;
        ints[1] = r - 1;
        return ints;

    }

    //搜索旋转排序数组
    //整数数组 nums 按升序排列，数组中的值 互不相同 。
    //
    //在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
    //
    //给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
    //
    //你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                //mid左边是升序
                if (nums[0] <= target && target < nums[mid]) {
                    //在左边升序的这边
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                //mid右边是升序
                if (nums[mid] < target && target <= nums[nums.length - 1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return -1;
    }

    //寻找两个正序数组的中位数
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length-1;

        while (left<right){
            int mid = (left+right)/2;

            if(nums[right]<nums[mid]){
                left = mid+1;
            }else {
                right=mid;
            }
        }

        return nums[left];
    }

    //有效的括号
    public boolean isValid(String s) {
        Stack<Character>stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c=='('||c=='{'||c=='['){
                stack.push(c);
            }else {
                if(c==')'){
                    if(stack.isEmpty()||!stack.pop().equals('(')){
                        return false;
                    }
                }
                if(c=='}'){
                    if(stack.isEmpty()||!stack.pop().equals('{')){
                        return false;
                    }
                }

                if(c==']'){
                    if(stack.isEmpty()||!stack.pop().equals('[')){
                        return false;
                    }
                }
            }
        }


        return stack.isEmpty();
    }

    //最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 0; i < text1.length(); i++) {
            dp[i][0] = 0;
        }

        for (int i = 0; i < text2.length(); i++) {
            dp[0][i] = 0;
        }

        for (int i = 1; i <= text1.length(); i++) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j <= text2.length(); j++) {
                char c2 = text2.charAt(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[text1.length()][text2.length()];
    }

    //旋转链表
    //给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null||k==0) {
            return head;
        }

        //计算链表长度
        int length=1;
        ListNode node = head;
        while (node.next!=null){
            length++;
            node = node.next;
        }

        //连接链表尾部
        node.next=head;

        //计算向右移动的次数n   <length
        int n = length-k%length;

        //定位新的head节点
        for (int i = 0; i < n-1; i++) {
            head = head.next;
        }

        ListNode curHead = head.next;
        //断开链表尾部
        head.next = null;
        return curHead;
    }

    //编辑距离
    //给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
    //
    //你可以对一个单词进行如下三种操作：
    //
    //插入一个字符
    //删除一个字符
    //替换一个字符
    public int minDistance(String word1, String word2) {
            int [][]dp = new int[word1.length()+1][word2.length()+1];
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= word2.length(); i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= word1.length(); i++) {
            char c1 = word1.charAt(i-1);
            for (int j = 1; j <=word2.length(); j++) {
                char c2 = word2.charAt(j-1);

                if(c1==c2){
                    dp[i][j] = Math.min(Math.min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]);
                }else {
                    dp[i][j] = Math.min(Math.min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+1);
                }
            }
        }

        return dp[word1.length()][word2.length()];
    }

    //和为 K 的连续子数组
    //返回所有和为K的连续子数组的个数
    public int subarraySum(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();//连续和，出现次数
        map.put(0,1);
        int sum = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            sum+=nums[i];
            if(map.containsKey(sum-k)){
                count += map.get(sum-k);
            }
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        return count;
    }


    public static void main(String[] args) throws InterruptedException {
        Solution solution = new Solution();
//        Ip ip = new Ip();
//        List<String> ips = ip.getIps("1231232123");
//        boolean b = ip.validIp4("12.23.34.12");
//        System.out.println(b);
//        System.out.println(ips);
//
//
//        MedianFinder medianFinder = new MedianFinder();
//        medianFinder.addNum(1);
//        medianFinder.addNum(9);
//        medianFinder.addNum(3);
//        medianFinder.addNum(2);
//        medianFinder.addNum(2);
//        System.out.println("中位数" + medianFinder.findMedian());
//
//
//        Counter counter = new Counter();
//        Integer[] element = {1, 22, 22, 22, 22, 1, 1, 22, 14, 12, 12, 12, 12};
//        List<Integer> integers = counter.topK(element, 10);
//        System.out.println("字符前K个频率最高" + integers);
//
//        int[] nums = {-1, 2, 1, -1, 2, -1, 2};
//        System.out.println("最大连续子数组" + counter.maxSubArray(nums));
//        System.out.println("查找最大连续元素个数" + counter.maxRepeat(element, 123));
//        System.out.println("最大连续元素" + counter.maxRepeat(element));
//        System.out.println("最大连续元素和个数" + counter.maxRepeatAndCount(element));
//        System.out.println("所有连续元素信息" + counter.repeatAndCount(element));
//        System.out.println("移除连续元素后" + counter.removeRepeat(element));
//
//        Permutation permutation = new Permutation();
//        Integer[] integers1 = {1, 2, 3};
//        Set<List<Integer>> rank = permutation.rank(integers1);
//        System.out.println(rank);
//        System.out.println(rank.size());
//
//
//        String s = "bcabvxcjpwrtup123213sfs";
//        String s1 = counter.removeDuplicate(s);
//        System.out.println(s1);
//
//        Shape shape = new Shape();
//        shape.diamond(11, true);
//
//        NQueens nQueens = new NQueens();
//        nQueens.print(10);
//        StringUtil stringUtil = new StringUtil();
//        String s = stringUtil.reverseBracketStr("(ed(et(oc)(od))el)");
//        System.out.println(s);

//        ListUtil.ListNode listNode1 = new ListUtil.ListNode(1);
//        ListUtil.ListNode listNode3 = new ListUtil.ListNode(3);
//        ListUtil.ListNode listNode5 = new ListUtil.ListNode(5);
//        listNode1.next = listNode3;
//        listNode3.next = listNode5;
//
//        ListUtil.ListNode listNode2 = new ListUtil.ListNode(2);
//        ListUtil.ListNode listNode4 = new ListUtil.ListNode(4);
//        ListUtil.ListNode listNode6 = new ListUtil.ListNode(6);
//        listNode2.next = listNode4;
//        listNode4.next = listNode6;
//
//        ListUtil.ListNode listNode = ListUtil.mergeList(listNode1, listNode2);
//        System.out.println(listNode);

//        Permutation permutation = new Permutation();
//        int[] temp = new int[]{1,2,3};
////        List<List<Integer>> permute = permutation.permute(temp);
////        System.out.println(permute);
//
//        Solution solution = new Solution();
//        List<List<Integer>> subsets = solution.subsets(temp);
//        System.out.println(subsets);

//        System.out.println(solution.longestPalindrome("babad"));

//        StringUtil stringUtil = new StringUtil();
//        List<String> strings = stringUtil.generateParenthesis(3);
//        System.out.println(strings);

//        Solution solution = new Solution();
//        System.out.println(solution.longestValidParentheses(")()())"));
//        int[] nums = new int[]{10,9,2,4,5,3,7};
//        System.out.println(solution.lengthOfLIS(nums));

//        Solution solution = new Solution();
//        System.out.println(solution.lengthOfLongestSubstring("pwwkew"));
//        System.out.println(solution.rank("123"));
//        System.out.println(solution.getBrackets(3));
//        System.out.println(solution.partition("123"));
//
//
//
//        int [] arr1 = new int[]{1,4,7,99,109};
//        int [] arr2 = new int[]{3,4,9,18,100};
//        System.out.println(Arrays.toString(solution.merger(arr1, arr2))
//        );
//
//        int [] arr3 = new int[]{-2,0,-1};
//        System.out.println(solution.maxProduct(arr3));

//        Solution solution = new Solution();
//        int [] arr4 = new int[]{3,2,3,19,6,4,8};
//        System.out.println(solution.findKthLargest(arr4, 2));
//
//
//        int [] arr1 = new int[]{1,4,7,99,109};
//        int [] arr2 = new int[]{3,4,9,18,100};
//        System.out.println(Arrays.toString(solution.merger(arr1, arr2))
//        );
//
//        int [] arr3 = new int[]{-2,0,-1};
//        System.out.println(solution.maxProduct(arr3));

//        char[][]grid = new char[][]{{'1','1','1'},{'0','1','0'},{'1','1','1'}};
//        System.out.println(solution.numIslands(grid));

//        System.out.println(solution.reverse(1534236469));
//        System.out.println(solution.letterCombinations("23"));
//        int [] arr= new int[]{1,2,3};
//        solution.nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));

//        int [] arr= new int[]{7,3,9,6};
//        System.out.println(solution.combinationSum(arr, 6));

//        List<String>list = new ArrayList<>();
//        list.add("car");
//        list.add("ca");
//        list.add("rs");
//        boolean leetcode = solution.wordBreak("cars", list);
//        System.out.println(leetcode);

//        int [] arr= new int[]{1,1,1,2,2,2,2,3,3,3,3,3};
//        int[] ints = solution.topKFrequent(arr, 1);
//        System.out.println(Arrays.toString(ints));
        char[][] chars = {{'A', 'B', 'C', 'D'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
        boolean abcced = solution.exist(chars, "ABCCED");

        int[][] ints = {{2, 1, 1}, {1, 1, 0}, {0, 1, 1}};
        int i = solution.orangesRotting(ints);

    }
}
