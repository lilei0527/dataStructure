package leetcode;


import java.util.*;

/**
 * @author lilei
 **/
@SuppressWarnings("unused")
public class Solution {

    //2023.3.3回顾

    //全排列
    public List<String>rank(String s){
        List<String>result = new ArrayList<>();
        char[] chars = s.toCharArray();
        rank(result,0,chars);
        return result;
    }

    private void rank(List<String>result,int start,char[] chars){
        if(start==chars.length){
            result.add(new String(chars));
            return;
        }

        for (int i = start; i < chars.length; i++) {
            swap(chars,start,i);
            rank(result,start+1,chars);
            swap(chars,start,i);
        }


    }

    private void swap(char[]chars,int i,int j){
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    //合法的括号匹配
    public List<String> getBrackets(int n){
        List<String>result = new ArrayList<>();
        getBrackets(n,0,0,result,new StringBuilder());
        return result;
    }

    public void getBrackets(int n,int leftCount,int rightCount,List<String>result,StringBuilder stringBuilder){
        if(leftCount+rightCount==2*n){
            result.add(stringBuilder.toString());
            return;
        }

        if(leftCount<n){
            stringBuilder.append("(");
            getBrackets(n,leftCount+1,rightCount,result,stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length()-1);
        }

        if(rightCount<leftCount){
            stringBuilder.append(")");
            getBrackets(n,leftCount,rightCount+1,result,stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length()-1);
        }
    }


    //反转括号内字符串
    private String reverse(String s){
        StringBuilder stringBuilder = new StringBuilder();
        Stack<String>stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c=='('){
                stack.push(stringBuilder.toString());
                stringBuilder.setLength(0);
            }else if(c==')'){
                stringBuilder.reverse();
                stringBuilder.insert(0,stack.pop());
            }else {
                stringBuilder.append(c);
            }
        }
        return stringBuilder.toString();
    }



    //找出最长有效（格式正确且连续）括号子串的长度
    private int getLength(String s){
        int max =0;
        Stack<Integer>stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c=='('){
                stack.push(i);
            }else {
                stack.pop();
                if(stack.isEmpty()){
                    stack.push(i);
                }else {
                    max = Math.max(max,i-stack.peek());
                }
            }
        }
        return max;
    }



    //最长回文子串
    private String getMaxRepeat(String s){
        int maxIndex=0;
        int maxLength = 0;
        for (int i = 0; i < s.length(); i++) {
            int maxLength1 = getMaxRepeat(i, i, s);
            int maxLength2 = getMaxRepeat(i, i+1, s);
            int maxLength3 = Math.max(maxLength1,maxLength2);
            if(maxLength3>=maxLength){
                maxIndex = i;
                maxLength = maxLength3;
            }
        }
        return s.substring(maxIndex-(maxLength-1)/2,maxIndex+maxLength/2+1);
    }


    private int getMaxRepeat(int leftIndex,int rightIndex,String s){
        while (leftIndex>=0&&rightIndex<s.length()){
            if(s.charAt(leftIndex)!=s.charAt(rightIndex)){
                break;
            }
            leftIndex--;
            rightIndex++;
        }
        return rightIndex-leftIndex-1;
    }

    /**
     * 给定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。
     *
     * 示例 1 :
     *
     * 输入: 2736
     * 输出: 7236
     * 解释: 交换数字2和数字7。
     * 示例 2 :
     *
     * 输入: 9973
     * 输出: 9973
     * 解释: 不需要交换。
     *
     */

    public int maximumSwap(int num) {
        String s = String.valueOf(num);
        int maxLeft=0;
        int maxRight=0;
        for (int i = 0; i < s.length(); i++){
            int maxChar = s.charAt(i)-'0';
            maxLeft = i;
            maxRight = i;
            for (int j = i + 1; j < s.length(); j++) {
                int rightChar = s.charAt(j) - '0';
                if (rightChar >= maxChar) {
                    maxChar = rightChar;
                    maxRight = j;
                }
            }
            if(s.charAt(i)-'0'!=s.charAt(maxRight) - '0'){//如果左边不是最大值,直接交换就好
                break;
            }
        }
        char[] chars = s.toCharArray();
        swap(chars,maxLeft,maxRight);
        String result = new String(chars);
        return Integer.parseInt(result);
    }

    //单调递增子序列长度
    public int getLength(int[] nums){
        if(nums.length==0){
            return 0;
        }
        int[]dp = new int[nums.length];
        dp[0]=1;
        int max = 1;
        for (int i = 1; i < nums.length; i++) {
            int dpMax = 1;
            for (int j = 0; j < dp.length; j++) {
                if(nums[i]>nums[j]){
                    dpMax = Math.max(dpMax,dp[j]+1);
                }
            }

            dp[i]=dpMax;
            max = Math.max(dpMax,max);
        }
        return max;
    }


    /**
     * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
     *
     * 回文串 是正着读和反着读都一样的字符串。
     *
     *
     * 示例 1：
     *
     * 输入：s = "aab"
     * 输出：[["a","a","b"],["aa","b"]]
     * 示例 2：
     *
     * 输入：s = "a"
     * 输出：[["a"]]
     *
     */
    public List<List<String>> partition(String s) {
        List<List<String>>result = new ArrayList<>();
        List<String>list = new ArrayList<>();

        int[][]arrs =new int[s.length()][s.length()];
        for (int i = 0; i < arrs.length; i++) {
            Arrays.fill(arrs[i],0);
        }
        dfs(s,0,list,result,arrs);
        return result;
    }
    
    
    private void dfs(String s,int start,List<String>list,List<List<String>>result,int[][]arrs){
        if(start==s.length()){
            result.add(new ArrayList<>(list));
        }



        for (int i = start; i < s.length(); i++) {
            if(isPalindrome(start,i,arrs,s)==1){
                list.add(s.substring(start,i+1));
                dfs(s,i+1,list,result,arrs);
                list.remove(list.size()-1);
            }
        }
    }

    private int isPalindrome(int i,int j,int[][]arrs,String s){
        if(arrs[i][j]!=0){
            return arrs[i][j];
        }

        if(i>=j){
            arrs[i][j] = 1;
        }else if(s.charAt(i)==s.charAt(j)){
            arrs[i][j] = isPalindrome(i+1,j-1,arrs,s);
        }else {
            arrs[i][j] = -1;
        }

        return arrs[i][j];
    }




    /**
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     *
     * 如果不存在公共前缀，返回空字符串 ""。
     */
    public String longestCommonPrefix(String[] strs) {
        if(strs.length==0){
            return "";
        }

        int length = strs[0].length();
        for (int i = 0; i < length; i++) {
            char c = strs[0].charAt(i);
            boolean isSame = true;
            for (int j = 0; j < strs.length; j++) {
                if(strs[j].length()==i||strs[j].charAt(i)!=c){
                    return strs[0].substring(0,i);
                }
            }

        }
        return strs[0];
    }



    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。
     *
     *
     * 输入: s = "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     * 示例 2:
     *
     * 输入: s = "bbbbb"
     * 输出: 1
     * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
     * 示例 3:
     *
     * 输入: s = "pwwkew"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是"wke"，所以其长度为 3。
     *     请注意，你的答案必须是 子串 的长度，"pwke"是一个子序列，不是子串。
     */
    public int lengthOfLongestSubstring(String s) {
        if(s==null||s.isEmpty()){
            return 0;
        }
        int preMax = 1;
        int max = 1;
        for (int i = 1; i < s.length(); i++) {
            String temp = s.substring(i-preMax,i);//前一个元素不重复的最长字符串
            char c = s.charAt(i);
            int index = temp.indexOf(c);//目前字符在前一个元素不重复的最长字符串d的最后索引
            if(index!=-1){
                preMax=temp.length()-index;
            }else {
                preMax=preMax+1;
            }
            max = Math.max(max,preMax);
        }
        return max;
    }

    /**
     *
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     *
     * 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     *
     * 示例 1：
     *
     * 输入：nums = [10,9,2,5,3,7,101,18]
     * 输出：4
     * 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
     * 示例 2：
     *
     * 输入：nums = [0,1,0,3,2,3]
     * 输出：4
     *
     *
     * 示例 3：
     *
     * 输入：nums = [7,7,7,7,7,7,7]
     * 输出：1
     *
     */


    public int lengthOfLIS(int[] nums) {
        List<Integer>list = new ArrayList<>();
        list.add(nums[0]);

        for (int i = 1; i < nums.length; i++) {
            if(nums[i]>list.get(list.size()-1)){
                list.add(nums[i]);
            }else {
                int left=0,right=list.size()-1,mid;
                while (left<=right){
                    mid = (left+right)/2;
                    if(list.get(mid)<nums[i]){
                        left=mid+1;
                    }else if(list.get(mid)>nums[i]){
                        right=mid-1;
                    }else {
                        left=mid;
                        break;
                    }
                }
                list.set(left,nums[i]);
            }
        }

        return list.size();
    }


    /**
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点
     */
    public ListUtil.ListNode removeNthFromEnd(ListUtil.ListNode head, int n) {
        ListUtil.ListNode quick = head;
        for (int i = 0; i < n-1; i++) {
            quick = quick.next;
        }
        ListUtil.ListNode slow = head;
        ListUtil.ListNode pre = null;

        while (quick.next!=null){
            quick = quick.next;
            pre = slow;
            slow = slow.next;
        }

        if(pre==null){
            return slow.next;
        }else {
            pre.next = slow.next;
            return head;
        }
    }

    /**
     * 给你一个只包含 '('和 ')'的字符串，找出最长有效（格式正确且连续）括号子串的长度。
     *
     *
     * 示例 1：
     *
     * 输入：s = "(()"
     * 输出：2
     * 解释：最长有效括号子串是 "()"
     * 示例 2：
     *
     * 输入：s = ")()())"
     * 输出：4
     * 解释：最长有效括号子串是 "()()"
     * 示例 3：
     *
     * 输入：s = ""
     * 输出：0
     *
     */
    public int longestValidParentheses(String s) {
        if(s==null||s.isEmpty()){
            return 0;
        }

        int max = 0;
        Stack<Integer>stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if(s.charAt(i)=='('){
                stack.push(i);
            }else {
                stack.pop();
                if(stack.isEmpty()){
                    stack.push(i);
                }else {
                    max = Math.max(max,i-stack.peek());
                }
            }
        }

        return max;
    }

    /**
     *
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
     *接雨水
     *
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水
     */

    public int trap(int[] height) {
        int result = 0;
        int left=0;
        int right=height.length-1;
        int leftMax=0;
        int rightMax=0;
        while (left<right){
            if(height[left]<height[right]){
                if(height[left]>leftMax){
                    leftMax=height[left];
                }else {
                    result+=leftMax-height[left];
                }
                left++;
            }else {
                if(height[right]>rightMax){
                    rightMax=height[right];
                }else {
                    result+=rightMax-height[right];
                }
                right--;
            }
        }
        return result;
    }

    /**
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     *
     * 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
     *
     *
     * 示例 1：
     *
     * 输入：s = "babad"
     * 输出："bab"
     * 解释："aba" 同样是符合题意的答案。
     * 示例 2：
     *
     * 输入：s = "cbbd"
     * 输出："bb"
     */

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }

        String result = "";
        int maxLen = 0;
        int maxIndex = 0;
        for (int i = 0; i < s.length(); i++) {
            int one = getPalindromeLen(s, i, i);
            int two = getPalindromeLen(s, i, i+1);
            int m = Math.max(one, two);
            if(m>maxLen){
                maxLen = m;
                maxIndex = i;
            }
        }
        return s.substring(maxIndex-(maxLen-1)/2,maxIndex+maxLen/2+1);
    }

    private int getPalindromeLen(String s,int left,int right){
        while (left>=0&&right<s.length()&&s.charAt(left)==s.charAt(right)){
            left--;
            right++;
        }
        return right-left-1;
    }



    /**
     * 幂集。编写一种方法，返回某集合的所有子集。集合中不包含重复的元素。
     *  输入： nums = [1,2,3]
     *  输出：
     *  [3],
     *  [1],
     *  [2],
     *  [1,2,3],
     *  [1,3],
     *  [2,3],
     *  [1,2],
     *  []
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        subsets(nums,0,list,result);
        return result;
    }

    public void subsets(int[] nums,int index,List<Integer> list,List<List<Integer>> result){
        if(index == nums.length){
            result.add(new ArrayList<>(list));
            return;
        }

        //选择index位置上的数
        list.add(nums[index]);
        subsets(nums,index+1,list,result);

        //不选择index位置上的数
        list.remove(list.size()-1);
        subsets(nums,index+1,list,result);

    }


    static class NQueens {

        public List<Map<Integer,Integer>> getNQueens(int n) {
            List<Map<Integer,Integer>>list = new ArrayList<>();
            int[] segment = new int[n];
            fill(0, n, segment,list);
            return list;
        }

        private void fill(int row, int n, int[] segment,List<Map<Integer,Integer>>list) {
            if (row == n) {
                Map<Integer,Integer>queens = new TreeMap<>();
                for (int i = 0; i < segment.length; i++) {
                    queens.put(i,segment[i]);
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
                fill(row + 1, n, segment,list);
            }
        }

        @SuppressWarnings("SameParameterValue")
        private void print(int n){
            List<Map<Integer, Integer>> nQueens = getNQueens(n);
            for(Map<Integer,Integer>map:nQueens) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if(j==map.get(i)){
                            System.out.print("Q");
                        }else {
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

        public static ListNode ReverseList(ListNode head) {
            if (head == null) return null;
            ListNode pre = null, next;
            while (head.next != null) {
                next = head.next;
                head.next = pre;
                pre = head;
                head = next;
            }
            head.next = pre;
            return head;
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
        public static ListNode mergeList(ListNode list1,ListNode list2){
            ListNode listNode = new ListNode(-1);
            ListNode head = listNode;
            while (list1!=null&&list2!=null){
                if(list1.val>=list2.val){
                    head.next = list2;
                    list2 = list2.next;
                }else {
                    head.next = list1;
                    list1 = list1.next;
                }
                head = head.next;
            }

            if(list1==null){
                head.next = list2;
            }else {
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
            List<List<Integer>> result= new ArrayList<>();
            backtrack(nums.length,ints,result,0);
            return result;
        }

        private void backtrack(int length,List<Integer>list,List<List<Integer>> result,int index){
            if(index==length){
                result.add(new ArrayList<>(list));
                return;
            }

            for (int i = index; i < length; i++) {
                Collections.swap(list,index,i);
                backtrack(length,list,result,index+1);
                Collections.swap(list,index,i);
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

    /**
     请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

     注意，您的结果中 不应 包含任何括号。
     示例 1：

     输入：s = “(abcd)”
     输出：“dcba”

     示例 2：

     输入：s = “(u(love)i)”
     输出：“iloveu”

     示例 3：

     输入：s = “(ed(et(oc))el)”
     输出：“leetcode”
     */
    public static class StringUtil{
        public String reverseBracketStr(String s){
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

    }


    public static void main(String[] args) {
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
        StringUtil stringUtil = new StringUtil();
        String s = stringUtil.reverseBracketStr("(ed(et(oc)(od))el)");
        System.out.println(s);

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

        Solution solution = new Solution();
        System.out.println(solution.lengthOfLongestSubstring("pwwkew"));
        System.out.println(solution.rank("123"));
        System.out.println(solution.getBrackets(3));
        System.out.println(solution.partition("123"));



    }
}
