import leetcode.Solution;

import java.util.*;

public class Hot {

    public static class ListNode {
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
        //出现了重复子串最大的位置
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

        //反转中点右边的链表
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
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        levelOrder1(Collections.singletonList(root), res);
        return res;
    }

    public void levelOrder1(List<TreeNode> nodes, List<List<Integer>> res) {
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
        levelOrder1(child, res);
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

    /**
     * lru缓存
     * 维护一个双向链表
     * 获取数据的时候把数据放到链表头
     * 存入数据的时候判断key是否存在，如果存在,替换value,将key放到链表头
     * 如果不存在，存入kv键值对，将key放到链表头，如果超过容量,移除链表末尾元素
     */
     class LRUCache {
         class Node{
            int key;
            int val;
            Node next;
            Node pre;

            public Node(int key, int val) {
                this.key = key;
                this.val = val;
            }
            public Node(){}
        }

        Map<Integer,Node> map = new HashMap<>();
        int capacity;
        int size=0;
        Node head ;
        Node tail ;

        //移除链表中的某个节点
        void removeNode(Node node){
            node.pre.next = node.next;
            node.next.pre = node.pre;
        }

        void addToHead(Node node){
            node.pre = head;
            node.next = head.next;
            head.next.pre = node;
            head.next = node;
        }

        void moveToHead(Node node){
            removeNode(node);
            addToHead(node);
        }

        Node removeFromTail(){
            Node pre = tail.pre;
            removeNode(tail.pre);
            return pre;
        }


        public LRUCache(int capacity) {
            this.capacity = capacity;
            head = new Node();
            tail = new Node();
            head.next = tail;
            tail.pre = head;
        }

        public int get(int key) {
            Node node = map.get(key);
            if(node==null){
                return -1;
            }else{
                moveToHead(node);
                return node.val;
            }
        }

        public void put(int key, int value) {
            Node node = map.get(key);
            if(node==null){
                Node newNode = new Node(key, value);
                map.put(key,newNode);
                addToHead(newNode);
                size++;
                if(size>capacity){
                    Node fromTail = removeFromTail();
                    map.remove(fromTail.key);
                    size--;
                }
            }else{
                node.val=value;
                moveToHead(node);
            }
        }
    }


    //二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
         List<List<Integer>> res = new ArrayList<>();
         if(root==null){
             return res;
         }
        zigzagLevelOrder(res,Collections.singletonList(root));
         return res;
    }

    public void zigzagLevelOrder(List<List<Integer>> res, List<TreeNode> treeNodes) {
        if (treeNodes.isEmpty()) {
            return;
        }
        List<TreeNode> children = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i <treeNodes.size(); i++) {
            TreeNode treeNode = treeNodes.get(i);
            list.add(treeNode.val);

            if (treeNode.left != null) {
                children.add(treeNode.left);
            }
            if (treeNode.right != null) {
                children.add(treeNode.right);
            }
        }
        if(res.size()%2!=0){
            Collections.reverse(list);
        }
        res.add(list);
        zigzagLevelOrder(res,children);
    }

    //螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int rowLength = matrix.length;
        int colLength = matrix[0].length;
        int size = rowLength*colLength;
        int circle=0;
        while (true){
            //左往右
            for (int i = circle; i < colLength-circle; i++) {
                res.add(matrix[circle][i]);
                if(res.size()==size){
                    return res;
                }
            }
            //上往下
            for (int i = circle+1; i < rowLength-circle; i++) {
                res.add(matrix[i][colLength-circle-1]);
                if(res.size()==size){
                    return res;
                }
            }
            //右往左
            for (int i = colLength-circle-2; i >=circle; i--) {
                res.add(matrix[rowLength-circle-1][i]);
                if(res.size()==size){
                    return res;
                }
            }
            //下往上
            for (int i = rowLength-circle-2; i >circle ; i--) {
                res.add(matrix[i][circle]);
                if(res.size()==size){
                    return res;
                }
            }
            circle++;
        }
    }

    //复原 IP 地址
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        List<String> list = new ArrayList<>();
        restoreIpAddresses(res,list,s,0);
        return res;
    }

    public void restoreIpAddresses(List<String>res,List<String>list,String s,int start){
        if(list.size()==4&&start==s.length()){
            StringBuilder sb = new StringBuilder();
            for (String string : list) {
                sb.append(string);
                sb.append(".");
            }
            sb.deleteCharAt(sb.length()-1);
            res.add(sb.toString());
            return;
        }

        //最多三位
        for (int i = start; i <start+3&&i<s.length() ; i++) {
            String sub = s.substring(start,i+1);
            //判断是否合法
            if(isLegal(sub)){
                list.add(sub);
                restoreIpAddresses(res,list,s,i+1);
                list.remove(list.size()-1);
            }
        }
    }

    public boolean isLegal(String s) {
        //是否是0-255的数字
        int i;
        try{
            i = Integer.parseInt(s);
        }catch(NumberFormatException e){
            return false;
        }
        //前导0
        if(s.charAt(0)=='0'&&s.length()>1){
            return false;
        }
        return i >= 0 && i <= 255;
    }

    //删除排序链表中的重复元素 II
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next=head;
        ListNode cur = dummy;
        while (cur.next != null&&cur.next.next!=null) {
            if(cur.next.val==cur.next.next.val){
                //cur.next = 第一个不重复的
                int repeat=cur.next.val;
                while(cur.next!=null&&cur.next.val==repeat){
                    cur.next = cur.next.next;
                }
            }else{
                cur=cur.next;
            }
        }
        return dummy.next;
    }

    //比较版本号
    public int compareVersion(String version1, String version2) {
        String[] split1 = version1.split("\\.");
        String[] split2 = version2.split("\\.");
        int max = Math.max(split1.length, split2.length);
        for (int i = 0; i < max; i++) {
            int i1=0;
            int i2=0;
            if(i<split1.length){
                i1 = Integer.parseInt(split1[i]);
            }
            if(i<split2.length){
                i2 = Integer.parseInt(split2[i]);
            }
            if(i1>i2){
                return 1;
            }else if(i1<i2){
                return -1;
            }
        }
        return 0;
    }

    //字符串相乘
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
         List<String>list=new ArrayList<>();
        for (int i = num1.length()-1; i >=0 ; i--) {
            int i1=num1.charAt(i)-'0';
            int jin=0;
            StringBuilder sb=new StringBuilder();
            for (int j = 0; j < num1.length()-1-i; j++) {
                sb.append('0');
            }
            for (int j = num2.length()-1; j >=0 ; j--) {
                int j1=num2.charAt(j)-'0';
                int sum=i1*j1+jin;
                int yu=sum%10;
                jin=sum/10;
                sb.append(yu);
            }
            if(jin!=0){
                sb.append(jin);
            }
            sb.reverse();
            list.add(sb.toString());
        }
        String s="";
        for (int i = 0; i < list.size(); i++) {
            s=add(s,list.get(i));
        }
        return s;
    }

    //字符串相加
    public String add(String num1, String num2) {
         int i=num1.length()-1;
         int j=num2.length()-1;
         int jin=0;
         StringBuilder sb=new StringBuilder();
         while (i>=0||j>=0||jin!=0){
            int i1=i>=0?num1.charAt(i)-'0':0;
            int j1=j>=0?num2.charAt(j)-'0':0;
            int sum=i1+j1+jin;
            int yu = sum%10;
            jin = sum/10;
            i--;
            j--;
            sb.append(yu);
         }
         sb.reverse();
         return sb.toString();
    }

    //字符串相乘
    public String multiply1(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int[] m = new int[num1.length() + num2.length()];
        for (int i = num1.length() - 1; i >= 0; i--) {
            int ci = num1.charAt(i) - '0';
            for (int j = num2.length() - 1; j >= 0; j--) {
                int cj = num2.charAt(j) - '0';
                m[i + j + 1] = m[i + j + 1] + ci * cj;
            }
        }


        int jin=0;
        for (int i = m.length-1; i >=0;i--) {
             jin = jin+m[i];
             m[i]=jin%10;
             jin=jin/10;
        }

        StringBuilder sb=new StringBuilder();
        int index = m[0] == 0 ? 1 : 0;
        for (int i = index; i < m.length; i++) {
            sb.append(m[i]);
        }
        return sb.toString();


    }


    //下一个排列
    public void nextPermutation(int[] nums) {
        for (int i = nums.length-2; i >=0 ; i--) {
            if(nums[i]<nums[i+1]){
                for (int j = nums.length-1; j >i ; j--) {
                    if(nums[j]>nums[i]){
                        swap(nums,i,j);
                        reverse(nums,i+1);
                        return;
                    }
                }
            }
        }
        reverse(nums,0);
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }

    public void swap(int[] nums, int i, int j) {
         int temp = nums[i];
         nums[i] = nums[j];
         nums[j] = temp;
    }


    //排序链表
    public ListNode sortList(ListNode head) {
         if(head==null||head.next==null){
             return head;
         }
        ListNode mid = getMid(head);
        ListNode nexHead = mid.next;
        mid.next=null;
       return mergeTwoLists(sortList(head),sortList(nexHead));
    }

    public ListNode getMid(ListNode node) {
        ListNode fast = node;
        ListNode slow = node;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            }else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 != null) {
            cur.next = l1;
        }else{
            cur.next=l2;
        }
        return dummy.next;
    }


    public static class LRU{
         class Node{
             int key;
             int val;
             Node next;
             Node pre;
             public Node(int key,int val){
                 this.key=key;
                 this.val=val;
             }

         }

         private Node head = new Node(0,0);
         private Node tail= new Node(0,0);
         private int capacity;
         private int size;

         private Map<Integer,Node> map = new HashMap<>();

         public LRU(int capacity) {
             this.capacity = capacity;
             head.next = tail;
             tail.pre = head;
             size = 0;
         }




         private void addToHead(Node node){
            node.next = head.next;
            head.next.pre = node;
            head.next = node;
            node.pre = head;
            size++;
            if(size>capacity){
                removeNode(tail.pre);
            }
         }


         private void removeNode(Node node){
             node.pre.next = node.next;
             node.next.pre = node.pre;
             size--;
             map.remove(node.key);
         }


         public void put(int key, int value) {
             Node node = map.get(key);
             if(node==null){
                 node = new Node(key,value);
                 map.put(key,node);
             }else{
                 node.val = value;
             }
             addToHead(node);
         }

         public int get(int key) {
             Node node = map.get(key);
             if(node==null){
                 return -1;
             }else{
                 removeNode(node);
                 addToHead(node);
                 return node.val;
             }
         }
    }


    public String longestPalindrome1(String s) {
        String res = "";
        for (int i = 0; i < s.length(); i++) {
            String s1 = longestPalindrome1(s, i, i);
            String s2 = longestPalindrome1(s, i, i+1);
            String s3 = s1.length() > s2.length()? s1 : s2;
            res = s3.length()>res.length()?s3:res;
        }
        return res;
    }

    public String longestPalindrome1(String s, int start, int end) {
        while (start >= 0 && end < s.length()) {
            if (s.charAt(start) != s.charAt(end)) {
                break;
            }
            start--;
            end++;
        }
        return s.substring(start+1,end);
    }
    
    //二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
         if(root==null){
             return Collections.emptyList();
         }
        List<List<Integer>> res = new ArrayList<>();
         List<TreeNode> treeNodes = new ArrayList<>();
        treeNodes.add(root);
        levelOrder(res,treeNodes);
        return res;
    }

    public void levelOrder(List<List<Integer>> res, List<TreeNode> treeNodes){
         if(treeNodes.isEmpty()){
             return;
         }
         List<Integer> list = new ArrayList<>();
         List<TreeNode> treeNodeList = new ArrayList<>();

        for (TreeNode treeNode : treeNodes) {
            list.add(treeNode.val);
            if (treeNode.left != null) {
                treeNodeList.add(treeNode.left);
            }
            if (treeNode.right != null) {
                treeNodeList.add(treeNode.right);
            }
        }
        res.add(list);
        levelOrder(res,treeNodeList);
    }


    //岛屿数量
    public int numIslands1(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    numIslands1(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    public void numIslands1(char[][] grid, int i, int j){
            if(i<0||i>=grid.length||j<0||j>=grid[0].length||grid[i][j]=='0'){
                return;
            }
            grid[i][j]='0';
        numIslands1(grid,i+1,j);
        numIslands1(grid,i-1,j);
        numIslands1(grid,i,j+1);
        numIslands1(grid,i,j-1);
    }

    //全排列
    public List<List<Integer>> permute1(int[] nums) {
         List<List<Integer>> res = new ArrayList<>();
         List<Integer> list = new ArrayList<>();
         for (int i = 0; i < nums.length; i++) {
             list.add(nums[i]);
         }
        permute1(res,list,0);
        return res;
    }

    public void permute1(List<List<Integer>> res, List<Integer> list,int start){
         if(start==list.size()){
             res.add(new ArrayList<>(list));
             return;
         }

        for (int i = start+1; i < list.size(); i++) {
            Collections.swap(list,start,i);
            permute1(res,list,start+1);
            Collections.swap(list,start,i);
        }
        }

        //二叉树的最近公共祖先
        public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
            Map<Integer, TreeNode> map = new HashMap<>();
            fill(root,map);

            Set<TreeNode> set = new HashSet<>();
            TreeNode nodeP = p;
            while(nodeP!=null){
                set.add(nodeP);
                nodeP=map.get(nodeP.val);
            }

            TreeNode nodeQ = q;
            while(nodeQ!=null){
                if(set.contains(nodeQ)){
                    return nodeQ;
                }
                nodeQ=map.get(nodeQ.val);
            }
            return null;

        }


        public void fill(TreeNode node,Map<Integer,TreeNode> map){
            if(node.left!=null){
                map.put(node.left.val,node);
                fill(node.left,map);
            }
            if(node.right!=null){
                map.put(node.right.val,node);
                fill(node.right,map);
            }
        }



        //最长公共子序列
        public int longestCommonSubsequence1(String text1, String text2) {
            int[][] dp = new int[text1.length()+1][text2.length()+1];
            for (int i = 0; i < text1.length(); i++) {
                for (int j = 0; j < text2.length(); j++) {
                    if(text1.charAt(i)==text2.charAt(j)){
                        dp[i+1][j+1] = dp[i][j]+1;
                    }else{
                        dp[i+1][j+1]=Math.max(dp[i+1][j],dp[i][j+1]);
                    }
                }
            }
            return dp[text1.length()][text2.length()];
        }


        //最长递增子序列
        public int lengthOfLIS1(int[] nums) {
            List<Integer>list = new ArrayList<>();
            list.add(nums[0]);
            for (int i = 1; i < nums.length; i++) {
                if(nums[i]>list.get(list.size()-1)){
                    list.add(nums[i]);
                    continue;
                }
                int left = 0;
                int right = list.size()-1;
                while (left<right){
                    int mid = (left+right)/2;
                    if(list.get(mid)<nums[i]){
                        left = mid+1;
                    }else if(list.get(mid)>nums[i]){
                        right = mid-1;
                    }else{
                        right = mid;
                    }
                }
                list.set(right,nums[i]);
            }
            return list.size();
        }



        //重排链表
        public void reorderList1(ListNode head) {
            //先找中点
            ListNode slow = head;
            ListNode fast = head;
            while(fast.next!=null&&fast.next.next!=null){
                slow = slow.next;
                fast = fast.next.next;
            }

            ListNode rightHead = slow.next;
            slow.next = null;

            //反转后半段链表
            ListNode pre = null;
            ListNode cur = rightHead;
            while(cur!=null){
                ListNode next = cur.next;
                cur.next = pre;
                pre = cur;
                cur = next;
            }

            //合并两个链表
            ListNode cur1 = head;
            ListNode cur2 = pre;
            while(cur1!=null&&cur2!=null){
                ListNode next1 = cur1.next;
                ListNode next2 = cur2.next;
                cur1.next = cur2;
                cur2.next = next1;
                cur1 = next1;
                cur2 = next2;
            }
        }


    public static void main(String[] args) {
        Hot hot = new Hot();
        System.out.println(hot.lengthOfLongestSubstring("tmmzuxt"));

        System.out.println(hot.multiply1("123", "456"));
        int[] num = {1,2,3};
        hot.nextPermutation(num);


        LRU lru = new LRU(2);
        lru.put(1,1);
        lru.put(2,2);
        lru.get(1);
        lru.put(3,3);
        lru.get(2);
        lru.put(4,4);
        lru.get(1);
        lru.get(3);
        lru.get(4);
    }
}

