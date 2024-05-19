import leetcode.Solution;

import java.util.*;

public class Hot {

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

    //无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            Integer index = map.get(c);
            if (index != null) {
                left = Math.max(left, index + 1);
            }
            max = Math.max(max, i - left + 1);
            map.put(c, i);
        }
        return max;
    }

    //最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        ;
        for (int i = 0; i < text1.length() + 1; i++) {
            dp[i][0] = 0;
        }
        for (int j = 0; j < text2.length() + 1; j++) {
            dp[0][j] = 0;
        }
        for (int i = 1; i <= text1.length(); i++) {
            char c = text1.charAt(i - 1);
            for (int j = 1; j <= text2.length(); j++) {
                if (c == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[text1.length()][text2.length()];
    }


    //反转链表
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }


    //最大子数组和
    public int maxSubArray(int[] nums) {
        int max = Integer.MIN_VALUE;
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum = Math.max(sum + nums[i], nums[i]);
            max = Math.max(max, sum);
        }
        return max;
    }

    //合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                curr.next = list1;
                list1 = list1.next;
            } else {
                curr.next = list2;
                list2 = list2.next;
            }
            curr = curr.next;
        }
        curr.next = list1 == null ? list2 : list1;
        return dummy.next;
    }

    //最长递增子序列
    public int lengthOfLIS(int[] nums) {
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > list.get(list.size() - 1)) {
                list.add(nums[i]);
            } else {
                int left = 0;
                int right = list.size() - 1;

                while (left <= right) {
                    int mid = (left + right) / 2;
                    if (nums[i] < list.get(mid)) {
                        right = mid - 1;
                    } else if (nums[i] > list.get(mid)) {
                        left = mid + 1;
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


    //零钱兑换
    //给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
    //计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
    //你可以认为每种硬币的数量是无限的。
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int j = 0; j < coins.length; j++) {
            for (int i = coins[j]; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        if (dp[amount] == amount + 1) {
            return -1;
        }
        return dp[amount];
    }

    //无重复字符的最长子串
    public int lengthOfLongestSubstring1(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            Integer index = map.get(c);
            if (index != null) {
                left = Math.max(left, index + 1);
            }
            max = Math.max(max, i - left + 1);
            map.put(c, i);
        }
        return max;
    }

    //重排链表
    public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }

        //找到中点
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode rightHead = slow.next;
        slow.next = null;

        //反转中点左边的链表
        ListNode prev = null;//新的头节点
        while (rightHead != null) {
            ListNode next = rightHead.next;
            rightHead.next = prev;
            prev = rightHead;
            rightHead = next;
        }

        //将左右两个链表交错合并
        ListNode left = head;
        ListNode right = prev;
        while (left != null && right != null) {
            ListNode leftNext = left.next;
            ListNode rightNext = right.next;

            left.next = right;
            right.next = leftNext;

            left = leftNext;
            right = rightNext;
        }
    }

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


    //二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        levelOrder(Collections.singletonList(root), res);
        return res;
    }

    public void levelOrder(List<TreeNode> nodes, List<List<Integer>> res) {
        if (nodes.isEmpty()) {
            return;
        }
        List<TreeNode> child = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        for (TreeNode node : nodes) {
            list.add(node.val);
            if (node.left != null) {
                child.add(node.left);
            }
            if (node.right != null) {
                child.add(node.right);
            }
        }
        res.add(list);
        levelOrder(child, res);
    }

    //最长回文子串
    public String longestPalindrome(String s) {
        String res = "";
        for (int i = 0; i < s.length(); i++) {
            String s1 = longestPalindrome(s, i, i);
            String s2 = longestPalindrome(s, i, i+1);
            String max = s1.length() > s2.length() ? s1 : s2;
            res = res.length() > max.length() ? res : max;
        }
        return res;
    }

    public String longestPalindrome(String s,int left,int right){
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left+1,right);
    }

    //岛屿数量
    public int numIslands(char[][] grid) {
        int result = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    result++;
                    numIsLands(grid, i, j);
                }
            }
        }
        return result;
    }

    public void numIsLands(char[][] grid,int x,int y) {
        if(x<0||y<0||x>=grid.length||y>=grid[0].length||grid[x][y]=='0') return;
        grid[x][y]='0';
        numIsLands(grid,x,y-1);
        numIsLands(grid,x,y+1);
        numIsLands(grid,x-1,y);
        numIsLands(grid,x+1,y);
    }

    //全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        for (int num : nums) {
            list.add(num);
        }
        permute(res,list,0);
        return res;
    }

    public void permute(List<List<Integer>> res,List<Integer>list,int start) {
        if(start==list.size()){
            res.add(new ArrayList<>(list));
            return;
        }

        for(int i=start;i<list.size();i++){
            Collections.swap(list,i,start);
            permute(res,list,start+1);
            Collections.swap(list,i,start);
        }
    }

    //合并区间
    public int[][] merge(int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        res.add(intervals[0]);
        for (int i = 0; i < intervals.length; i++) {
            int[] last = res.get(res.size() - 1);
            if(intervals[i][0]<=last[1]){
                last[1]=Math.max(last[1],intervals[i][1]);
            }else{
                res.add(intervals[i]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    //删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = head;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;
        return dummy.next;
    }

    //数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k){
        return findKthLargest(nums,nums.length-k,0,nums.length-1);
    }

    public int findKthLargest(int[] nums, int index,int left,int right){
        int baseLeft = left;
        int baseRight = right;

        while(left<right){
            while(left<right&&nums[right]>=nums[baseLeft]){
                right--;
            }
            while (left<right&&nums[left]<=nums[baseLeft]){
                left++;
            }
            int temp = nums[left];
            nums[left]=nums[right];
            nums[right]=temp;
        }

        int temp = nums[baseLeft];
        nums[baseLeft]=nums[left];
        nums[left]=temp;

        if(left==index){
            return nums[index];
        }

        if(left>index){
            return findKthLargest(nums,index,baseLeft,left-1);
        }else{
            return findKthLargest(nums,index,left+1,baseRight);
        }
    }

    //三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if(i>0&&nums[i]==nums[i-1]){
                continue; // 第一个元素和第二个元素去除重复
            }

            int left = i+1;
            int right = nums.length-1;
            while(left<right){
                if(left!=i+1&&nums[left]==nums[left-1]){
                    left++;
                    continue;// 第二个元素和第三个元素去除重复
                }
                int sum = nums[i]+nums[left]+nums[right];
                if(sum>0){
                    right--;
                }else if(sum<0){
                    left++;
                }else{
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    res.add(list);
                    right--;
                    left++;
                }
            }

        }
        return res;
    }

    //搜索旋转排序数组
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while(left<=right){
            int mid = (left+right)/2;
            if(nums[mid]==target){
                return mid;
            }
            if(nums[mid]>=nums[0]){//中间节点左侧递增
                if(nums[0]<=target&&target<nums[mid]){ //要么是左侧，要么右侧
                    right=mid-1;
                }else{
                    left = mid+1;
                }
            }else{ //中间节点右侧递增
                if(nums[mid]<target&&target<=nums[nums.length-1]){
                    left=mid+1;
                }else{
                    right = mid-1;
                }
            }
        }
        return -1;
    }



    public static void main(String[] args) {

        Hot hot = new Hot();
        System.out.println(hot.lengthOfLongestSubstring("tmmzuxt"));
    }
}
