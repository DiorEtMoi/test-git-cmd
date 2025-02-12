package services;

import model.ListNode;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class LeetCode implements ILeetCode {
    //https://leetcode.com/problems/roman-to-integer
    public int romanToInt(String s) {
        Map<Character, Integer> romanMap = new HashMap<>();
        romanMap.put('I', 1);
        romanMap.put('V', 5);
        romanMap.put('X', 10);
        romanMap.put('L', 50);
        romanMap.put('C', 100);
        romanMap.put('D', 500);
        romanMap.put('M', 1000);
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if(i + 1 < s.length()
                    && romanMap.get(s.charAt(i + 1)) > romanMap.get(s.charAt(i))) {
                result += romanMap.get(s.charAt(i + 1)) - romanMap.get(s.charAt(i));
                i++;
            }else {
                result += romanMap.get(s.charAt(i));
            }
        }
        return result;
    }
    //https://leetcode.com/problems/longest-common-prefix
    @Override
    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0) return "";
        if(strs.length == 1) return strs[0];
        String s = strs[0];
        //check each of string in array every element and return when meet the different character
        for(int i = 0; i < s.length(); i++) {
            for(int j = 1; j < strs.length; j++) {
                if(strs[j].length() <= i || s.charAt(i) != strs[j].charAt(i)) {
                    return s.substring(0, i);
                }
            }
        }
        return s;
    }
    //https://leetcode.com/problems/valid-parentheses
    @Override
    public boolean isValid(String s) {
        // using stack to contain the open brackets
        Stack<Character> c = new Stack<>();
        if(s == null || s.isEmpty()) return false;
        if(s.length() == 1) return false;
        for (int i = 0; i < s.length(); i++) {
            char c1 = s.charAt(i);
            if(c1 == '(' || c1 == '{' || c1 == '[') {
                c.push(c1);
            }else if(c.isEmpty() || !match(c.pop(), c1)) {
                return false;
            }
        }
        return c.isEmpty();
    }

    @Override
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        //create dummy node to store the data
        // the current node to traverse of all node
        ListNode dummyNode = new ListNode();
        ListNode curr = dummyNode;
        while(list1 != null && list2 != null) {
            if(list1.val <= list2.val) {
                curr.next = list1;
                list1 = list1.next;
            } else {
                curr.next = list2;
                list2 = list2.next;
            }
            curr = curr.next;
        }
        return dummyNode.next;
    }
    // https://leetcode.com/problems/remove-duplicates-from-sorted-array
    @Override
    public int removeDuplicates(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if(i < nums.length - 1 && nums[i] == nums[i + 1]) {
                continue;
            }
            nums[count] = nums[i];
            count++;
        }
        return count;
    }

    @Override
    public int removeElement(int[] nums, int val) {
       int count = 0;
       for (int i = 0; i < nums.length; i++) {
           if(nums[i] != val) {
               nums[count++] = nums[i];
           }
       }
       return count;
    }

    @Override
    public int strStr(String haystack, String needle) {
        int m = haystack.length();
        int n = needle.length();
        for(int i = 0; i < m - n + 1; i++) {
            String subString = haystack.substring(i, i + n);
            if(Objects.equals(needle, subString)) {
                return i;
            }
        }
        return -1;
    }
    // https://leetcode.com/problems/search-insert-position/description/
    @Override
    public int searchInsert(int[] nums, int target) {
        for(int i = 0; i < nums.length; i++){
            if(nums[i] >= target){
                return i;
            }
        }
        return nums.length;
    }

    @Override
    public int lengthOfLastWord(String s) {
        String[] arrayString = s.split(" ");


        return arrayString[arrayString.length - 1].length();
    }

    @Override
    public int[] plusOne(int[] digits) {
       for(int i = digits.length - 1; i >= 0; i--){
           digits[i]++;
           digits[i] %= 10;
           if(digits[i] != 0){
                return digits;
           }
       }
       int[] result = new int[digits.length + 1];
       result[0] = 1;
        return result;
    }

    @Override
    public String addBinary(String a, String b) {

        return "";
    }

    @Override
    public int mySqrt(int x) {
        int left = 0;        // Initialize the left boundary of the search space
        int right = x;       // Initialize the right boundary of the search space

        while (left < right) { // Loop until the search space is narrowed down to one element
            int mid = (left + right + 1) >>> 1; // Compute the middle point, using unsigned right shift for safe division by 2

            if (mid <= x / mid) { // If the square of mid is less than or equal to x
                left = mid;        // Move the left boundary to mid, as mid is a potential solution
            } else {
                right = mid - 1;   // Otherwise, discard mid and the right search space
            }
        }
        // The loop exits when left == right, which will be the largest integer less than or equal to the sqrt(x)
        return left; // Return the calculated square root
    }

    @Override
    public int climbStairs(int n) {
        int[] memo = new int[n + 1];
        memo[0] = 1;
        memo[1] = 1;
        for(int i = 2; i <= n; i++) {
            memo[i] = memo[i - 1] + memo[i - 2];
        }
        return memo[n];
    }

    @Override
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        for(int row = 0; row < numRows; row++) {
            List<Integer> temp = new ArrayList<>();
            for(int col = 0; col <= row; col++){
                if(col == 0 || row == col){
                    temp.add(1);
                } else {
                    temp.add(result.get(row - 1).get(col - 1) +
                            result.get(row - 1).get(col));
                }
            }
            result.add(temp);
        }
        return result;
    }

    @Override
    public LinkedListNode nthToLast(LinkedListNode head, int n) {
        if (head == null || n < 1) {
            return null;
        }

        LinkedListNode p1 = head;
        LinkedListNode p2 = head;
        for (int j = 0; j < n - 1; ++j) { // p2 chạy hơn p1 n-1 phần tử
            if (p2 == null) return null; // Không tìm ra vì mảng ít hơn n phần tử
            p2 = p2.next;
        }

        //Cho 2 con trỏ cùng chạy tới cuối list
        while (p2.next != null) {
            p1 = p1.next;
            p2 = p2.next;
        }
        return p1;
    }

    @Override
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if(cur.val == cur.next.val){
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return head;

    }

    @Override
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int indexNums1 = m - 1;
        int indexNums2 = n - 1;
        int mergeSort = m + n - 1;
        while(indexNums2 >= 0){
            if(indexNums1 < 0 || nums1[indexNums1] <= nums2[indexNums2]){
                nums1[mergeSort] = nums2[indexNums2];
                indexNums2--;
            } else {
                nums1[mergeSort] = nums1[indexNums1];
                indexNums1--;
            }
            mergeSort--;
        }
    }

    @Override
    public List<Integer> getRow(int rowIndex) {
        List<List<Integer>> result = new ArrayList<>();
        for(int row = 0; row < rowIndex + 1; row++){
            List<Integer> temp = new ArrayList<>();
            for(int column = 0; column < row + 1; column++){
                if(column == 0 || row == column){
                    temp.add(1);
                } else {
                    temp.add(result.get(row - 1).get(column - 1) +
                            result.get(row - 1).get(column));
                }
            }
            result.add(temp);
        }
        return result.get(rowIndex);
    }

    @Override
    public int maxProfit(int[] prices) {
        int result = 0;
        int min = prices[0];
        for(int i = 1; i < prices.length; i++){
            result = Math.max(result, prices[i] - min);
            if(prices[i] < min){
                min = prices[i];
            }
        }
        return result;
    }

    @Override
    public boolean isPalindrome(String s) {
       int head = 0;
       int tail = s.length() - 1;
       while (head < tail) {
           if(!Character.isLetterOrDigit(s.charAt(head))){
               head++;
           } else if (!Character.isLetterOrDigit(s.charAt(tail))) {
               tail--;
           } else if (Character.toLowerCase(s.charAt(head)) != Character.toLowerCase(s.charAt(tail))) {
               return false;
           } else {
               head++;
               tail--;
           }
       }
       return true;
    }

    @Override
    public long maximumSubarraySum(int[] nums, int k) {
        int curSum = 0;
        Map<Integer,Integer> countMap = new HashMap<>(k);
        for(int i =0 ; i < k; i++){
            countMap.merge(nums[i], 1, Integer::sum);
            curSum = nums[i] + curSum;
        }
        int maxSum = countMap.size() == k ? curSum : 0;
        for(int i = k; i < nums.length; i++){
            countMap.merge(nums[i], 1, Integer::sum);
            curSum += nums[i] - nums[i - k];

            int count = countMap.merge(nums[i - k], -1, Integer::sum);
            if(count == 0){
                countMap.remove(nums[i - k]);
            }
            if(countMap.size() == k){
                maxSum = Math.max(maxSum, curSum);
            }
        }
        return maxSum;
    }

    @Override
    public int smallestSubWithSum(int[] arr, int n, int x) {
        int minSubArray = n + 1;
        for(int start = 0; start < n; start++){
            int curSum = arr[start];
            for(int end = start + 1; end < n; end++){
                curSum += arr[end];
                if(curSum > x && end - start + 1 < minSubArray){
                    minSubArray = end - start + 1;
                    break;
                }
            }
        }
        return minSubArray;
    }

    @Override
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> countMap = new HashMap<>();
        countMap.merge(nums[0], 1, Integer::sum);
        for(int i = 1; i < nums.length; i++){
            countMap.merge(nums[i], 1, Integer::sum);
        }
        for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
                if(entry.getValue() == 1){
                    return entry.getKey();
                }
        }
        return 0;
    }

    @Override
    public boolean hasCycle(ListNode head) {
        ListNode start = head;
        ListNode end = head;
        while(end != null && end.next != null){
            start = start.next;
            end = end.next.next;
            if(start == end){
                return true;
            }
        }
        return false;
    }

    @Override
    public int countUnguarded(int m, int n, int[][] guards, int[][] walls) {
        int[][] map = new int[m][n];
        for(int[] guard : guards){
            map[guard[0]][guard[1]] = 2;
        }
        for(int[] wall : walls){
            map[wall[0]][wall[1]] = 2;
        }
        int[] dirs = {-1,0,1,0,-1};
        for(int[] guard : guards){
            for(int i = 0; i < 4; i++){
                int x = guard[0];
                int y = guard[1];
                int deltaX = dirs[i];
                int deltaY = dirs[i + 1];
                while(x + deltaX >= 0 && x + deltaX < m &&
                y + deltaY >= 0 && y + deltaY < n && map[x + deltaX][y + deltaY] < 2){
                    x += deltaX;
                    y += deltaY;
                    map[x][y] = 1;
                }
            }
        }
        int count = 0;
        for(int[] i : map){
            for (int j : i){
                if(j == 0){
                    count++;
                }
            }
        }
        return count;
    }

    @Override
    public int[] decrypt(int[] code, int k) {
        int[] res = new int[code.length];
        for(int i = 0 ; i < 3; i++){
            res[code.length - 1] += code[i];
        }
        for(int i = 0; i < code.length - 1; i++){
            for(int j = 0; j < code.length; j++){

            }
        }
        return new int[0];
    }

    @Override
    public long maxMatrixSum(int[][] matrix) {
        long sum = 0;
        int minAbs = Integer.MAX_VALUE;
        int count = 0;
        for(int[] row : matrix){
            for( int col : row){
                sum += Math.abs(col);
                minAbs = Math.min(minAbs, Math.abs(col));
                if(col < 0){
                    count++;
                }
            }
        }
        if(count % 2 == 0 || minAbs == 0){
            return sum;
        }
        return sum - 2*minAbs;
    }

    @Override
    public int shortestSubarray(int[] nums, int k) {
        int min = Integer.MAX_VALUE;
        for(int i = 0 ; i < nums.length; i++){
            int sum = 0;
            for (int j = 0; j < nums.length; j++){
                sum += Math.abs(nums[j]);
                if(sum == k){
                    min = Math.min(min, j + 1);
                    break;
                }
            }
        }
        if(min == Integer.MAX_VALUE){
            return -1;
        }
        return min;
    }

    @Override
    public boolean rotateString(String s, String goal) {
        if(s.length() != goal.length()){
            return false;
        }
        String newString = s + s;

        return newString.contains(goal);
    }

    @Override
    public int majorityElement(int[] nums) {
        Map<Integer,Integer> mapCount = new HashMap<>();
        mapCount.merge(nums[0], 1, Integer::sum);
        for(int i = 1; i < nums.length; i++){
            mapCount.merge(nums[i], 1, Integer::sum);
        }
        int maxValue = 0;
        int res = -1;
        for (Map.Entry<Integer, Integer> entry : mapCount.entrySet()) {
            if(entry.getValue() > maxValue){
                maxValue = entry.getValue();
                res = entry.getKey();
            }
        }
        if(maxValue >= nums.length / 2){
            return res;
        }
        return 0;
    }

    @Override
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummyNode = new ListNode(-1);
        dummyNode.next = head;
        ListNode previousNode = dummyNode;
        while(previousNode.next != null){
            if(previousNode.next.val == val){
                previousNode.next = previousNode.next.next;
            } else {
                previousNode = previousNode.next;
            }
        }
        return dummyNode.next;
    }

    @Override
    public int countPrimes(int n) {
        boolean[] isPrime = new boolean[n];
        Arrays.fill(isPrime, true);
        int count = 0;
        for(int i = 2; i < n; i++){
            if(isPrime[i]){
                count++;
                for(int j = i * 2; j < n; j += i){
                    isPrime[j] = false;
                }
            }
        }
        return count;
    }

    @Override
    public boolean isHappy(int n) {
        if(n <= 0){
            return false;
        }
        while(n >= 10){
            n = calculateNumber(n);
        }
        if(n == 1){
            return true;
        }
        return false;
    }

    @Override
    public String makeFancyString(String s) {
        StringBuilder res = new StringBuilder();
        for(char c : s.toCharArray() ){
            int length = res.length();
            if(length > 1 && res.charAt(length - 1) == c &&
                    res.charAt(length - 2) == c){
                continue;
            }
            res.append(c);
        }
        return res.toString();
    }

    @Override
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> mapping = new HashMap<>();
        if(s.length() != t.length()){
            return false;
        }
        for(int i = 0; i < s.length(); i++){
            if(!mapping.containsKey(s.charAt(i))){
                if(mapping.containsValue(t.charAt(i))){
                    return false;
                }
                mapping.put(s.charAt(i), t.charAt(i));
            }else {
                if(t.charAt(i) != mapping.get(s.charAt(i))){
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public ListNode reverseList(ListNode head) {
       ListNode dummyNode = new ListNode();
       ListNode current = head;
       while(current != null){
           ListNode next = current.next;
            current.next = dummyNode;
            dummyNode.next = current;
            current = next;
       }
       return dummyNode.next;
    }

    @Override
    public boolean containsDuplicate(int[] nums) {
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int n : nums){
            if(countMap.containsKey(n)){
                return false;
            }
            countMap.put(n, countMap.getOrDefault(n, 0) + 1);
        }
        return false;
    }

    @Override
    public boolean isPowerOfTwo(int n) {
        if(n == 1){
            return true;
        }
        while(n != 1){
            if(n % 2 != 0){
                return false;
            }
            n /= 2;
        }
        return true;
    }

    @Override
    public int missingNumber(int[] nums) {
        int[] newA = Arrays.stream(nums).sorted().toArray();
        for(int i = 0; i < newA.length - 1; i++){
            if(Math.abs(newA[i] - newA[i + 1]) == 2){
                return newA[i] + 1;
            }
        }
        if(nums.length == 1){
            if(nums[0] == 1){
                return 0;
            }
            return 1;
        }
        if(nums[0] != 0){
            return 0;
        }
        return nums.length;
    }

    @Override
    public int reverse(int x) {
        int res = 0;
        while (x != 0){
            if (res < Integer.MIN_VALUE / 10 || res > Integer.MAX_VALUE / 10) {
                return 0;
            }
            res = res * 10 + x % 10;
            x /= 10;
        }
        return res;
    }

    @Override
    public String intToRoman(int num) {
       StringBuilder res = new StringBuilder();
        String[] romans = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        int[] values = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
        for(int i = 0; i < romans.length; i++){
            while(num >= values[i]){
                num -= values[i];
                res.append(romans[i]);
            }
        }
        return res.toString();
    }

    @Override
    public int coinChange(int[] coins, int amount) {
        int count = 0 ;
        for(int i = coins.length - 1; i >= 0; i--){
            if (amount >= coins[i]){
                count++;
                amount -= coins[i];
            }
        }
        return count;
    }

    @Override
    public boolean checkIfExist(int[] arr) {
        Set<Integer> set = new HashSet<>();
        for(int i : arr){
           if(set.contains(i * 2) || (set.contains(i / 2) && i % 2 == 0)){
               return true;
           }
           set.add(i);
        }
        return false;
    }

    @Override
    public int isPrefixOfWord(String sentence, String searchWord) {
        String[] splitString = sentence.split(" ");
        for(int i = 0; i < splitString.length; i++){
            if(splitString[i].startsWith(searchWord)){
                return i + 1;
            }
        }
        return -1;
    }

    @Override
    public String addSpaces(String s, int[] spaces) {
        StringBuilder res = new StringBuilder();
        int start = 0;
        for(int i = 0; i < spaces.length; i++){
            res.append(s, start, spaces[i])
                    .append(' ');
            start = spaces[i];
        }
        res.append(s,start,s.length());
        return res.toString();
    }

    @Override
        public int lengthOfLongestSubstring(String s) {
        int leftPointer = 0;
        int maxLength = 0;
        Set<Character> set = new HashSet<>();
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            while(set.contains(c)){
                set.remove(s.charAt(leftPointer++));
            }
            set.add(c);
            maxLength = Math.max(maxLength, i - leftPointer + 1);
        }
        return maxLength;
    }

    @Override
    public boolean canMakeSubsequence(String str1, String str2) {
        int currentIndex = 0;
        int str2Length = str2.length();
        for(char c : str1.toCharArray()){
            char nextChar = c == 'z' ? 'a' : (char) (c + 1);
            if(currentIndex < str2Length && (str2.charAt(currentIndex) == nextChar || str2.charAt(currentIndex) == c)){
                currentIndex++;
            }
        }
       return str2Length == currentIndex;
    }

    @Override
    public String longestPalindrome(String s) {
        String res = "";
        boolean match = false;
        for(int start = 0; start < s.length() - 1; start++){
            for(int end = start + 1; end < s.length(); end++){
                if(s.charAt(start) == s.charAt(end)){
                    String currentEnd = s.substring(start, end + 1);
                    if(res.isEmpty() || res.length() < currentEnd.length()){
                        res = currentEnd;
                        match = true;
                        break;
                    }
                }
            }
        }
        if(!match){
            return s.substring(0, 1);
        }
        return res;
    }


    // hand this problems using two pointers 1 start head of list and 2 will be start at the end of list
    // calculated the area and compare with the max before
    @Override
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxArea = 0;
        while(left < right){
            int currentArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currentArea);

            if(height[left] < height[right]){
                left++;
            }else{
                right--;
            }
        }
        return maxArea;
    }

    @Override
    public int maxSubArray(int[] nums) {
       int maxSum = nums[0];
       int leftIndex = 0;
       for(int i = 0; i < nums.length; i++){
           int currentSum = nums[i];
           while(nums[leftIndex] < 0){
               leftIndex++;
               continue;
           }
           if(currentSum + nums[i] < 0){
               leftIndex++;
               continue;
           }
           currentSum += nums[i];
           maxSum = Math.max(currentSum,maxSum);
       }
        return maxSum;
    }

    @Override
    public boolean canChange(String start, String target) {
        List<int[]> startPos = parseString(start);
        List<int[]> targetPos = parseString(target);
        if(startPos.size() != targetPos.size()){
            return false;
        }
        for(int i = 0; i < startPos.size(); i++){
            int[] startList = startPos.get(i);
            int[] targetList = targetPos.get(i);

            if(startList[0] != targetList[0]){
                return false;
            }
            if(startList[0] == 1 && startList[1] < targetList[1]){
                return false;
            }
            if(startList[0] == 0 && startList[1] > targetList[1]){
                return false;
            }
        }
        return true;
    }

    @Override
    public int maxCount(int[] banned, int n, int maxSum) {
        Set<Integer> setNums = new HashSet<>();
        int count = 0;
        for(int num : banned){
            setNums.add(num);
        }
        for(int i = 1; i <= n; i++){
           if(setNums.contains(i)){
                continue;
           }
           if(maxSum - i < 0){
               return count;
           }
           maxSum -= i;
           count++;
        }
        return count;
    }

    @Override
    public String compressedString(String word) {
        StringBuilder res = new StringBuilder();
       for(int i = 0, j = 0; i < word.length(); i = j){
           int count = 0;
           while (j < word.length() && word.charAt(j) == word.charAt(i) && count < 9){
               j++;
               count++;
           }
           res.append(count).append(word.charAt(i));
       }
        return res.toString();
    }

    @Override
    public boolean canMeasureWater(int x, int y, int target) {
        if(x + y < target){
            return false;
        }
        return target % GCD(x, y) == 0;
    }

    @Override
    public int maximumBeauty(int[] nums, int k) {
        int maxValue = Arrays.stream(nums).max().getAsInt() + 2 * k + 2;
        int[] delta = new int[maxValue];
        for(int i : nums){
            delta[i]++;
            delta[i + 2 * k + 1]--;
        }
        int sum = 0;
        int max = 0;
        for(int i : delta){
            sum += i;
            max = Math.max(max, sum);
        }
        return max;
    }

    @Override
    public long pickGifts(int[] gifts, int k) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        for(int i : gifts){
            maxHeap.offer(i);
        }
        for(int i = 0; i < k; i++){
            int maxNum = maxHeap.poll();
            int sqrt = (int) Math.sqrt(maxNum);
            maxHeap.offer(sqrt);
        }
        long res = 0;
        while(maxHeap.size() > 0){
            res += maxHeap.poll();
        }
        return res;
    }

    @Override
    public int[] finalPrices(int[] prices) {
        for(int i = 0; i < prices.length - 1; i++){
            for(int j = i + 1; j < prices.length; j++){
                if(prices[i] >= prices[j]){
                    prices[i] = prices[i] - prices[j];
                    break;
                }
            }
        }
        return prices;
    }

    @Override
    public boolean hasDuplicate(int[] nums) {
        Map<Integer, Integer> mapCount = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(mapCount.containsKey(nums[i])){
                return true;
            }
            mapCount.put(nums[i], mapCount.getOrDefault(nums[i], 0) + 1);
        }
        return false;
    }

    @Override
    public boolean isAnagram(String s, String t) {
        if(s.length() != t.length()){
            return false;
        }
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();
        Arrays.sort(sChar);
        Arrays.sort(tChar);
       return Arrays.equals(sChar, tChar);
    }

    @Override
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> mapValue = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            int value = target - nums[i];
            if(!mapValue.containsKey(value)){
                mapValue.put(value, i);
            }else{
                return new int[]{mapValue.get(value), i};
            }
        }
        return new int[0];
    }

    @Override
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for(String s : strs){
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if(!map.containsKey(key)){
                map.put(key, new ArrayList<>(Arrays.asList(s)));
            }else{
                map.get(key).add(s);
            }
        }
        List<List<String>> result = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            result.add(entry.getValue());
        }
        result = result.stream()
                .sorted(Comparator.comparingInt(List::size)) // Sort by list size
                .collect(Collectors.toList());
        return result;
    }

    @Override
    public int[] topKFrequent(int[] nums, int k) {
       Map<Integer,Integer> countMap = new HashMap<>();
       for(int i : nums){
           countMap.put(i, countMap.getOrDefault(i, 0) + 1);
       }
       List<int[]> result = new ArrayList<>();
       for(Map.Entry<Integer, Integer> entry : countMap.entrySet()){
            result.add(new int[]{entry.getKey(), entry.getValue()});
       }
       result.sort((a, b) -> b[1] - a[1]);
       int[] a = new int[k];
       for(int i = 0; i < k; i++){
           a[i] = result.get(i)[0];
       }
       return a;
    }

    @Override
    public int[] productExceptSelf(int[] nums) {
        int[] result = new int[nums.length];
        for(int i = 0; i < result.length; i++){
            int value = 1;
            for(int j = 0; j < nums.length; j++){
                if(i == j){
                    continue;
                }
                value *= nums[j];
            }
            result[i] = value;
        }
        return result;
    }

    @Override
    public int[] runningSum(int[] nums) {
        int[] res = new int[nums.length];
        int currentValue = 0;
        for(int i = 0; i < nums.length; i++){
            currentValue += nums[i];
            res[i] = currentValue;
        }
        return res;
    }

    @Override
    public int maximumWealth(int[][] accounts) {
        int maxValue = 0;
        for(int[] account : accounts){
            int currentWeath = 0;
            for(int num : account){
                currentWeath += num;
            }
            maxValue = Math.max(maxValue, currentWeath);
        }
        return maxValue;
    }

    @Override
    public String mergeAlternately(String word1, String word2) {
        String[] word1Array = word1.split("");
        String[] word2Array = word2.split("");
        String[] mergeArray = new String[word1Array.length + word2Array.length];
        Arrays.fill(mergeArray, "");
        int first = 0 ;
        int second = 0 ;
        for(int i = 0; i < mergeArray.length; i++){
            if(i % 2 == 0 && first < word1Array.length){
                mergeArray[i] = word1Array[first++];
            }else if (i % 2 != 0 && second < word2Array.length){
                mergeArray[i] = word2Array[second++];
            }
        }
        return String.join("",mergeArray);
    }

    @Override
    public int maxProfitV2(int[] prices) {
        int res = 0;
        int min = prices[0];
        for(int i = 1; i < prices.length; i++){
            int currentValue = prices[i] - min;
            if(currentValue > 0){
                res += currentValue;
                min = prices[i];
            }
            min = Math.min(min, prices[i]);
        }
        return res;
    }

    @Override
    public void rotate(int[] nums, int k) {
        int[] res = new int[nums.length];
        for(int i = 0; i < nums.length; i++){
            res[(i + k) % nums.length] = nums[i];
        }
        for(int i = 0; i < res.length; i++){
            nums[i] = res[i];
        }
    }

    @Override
    public int prefixCount(String[] words, String pref) {
        int count = 0;
        for(String w : words){
           if(w.startsWith(pref)){
               count++;
           }
        }
        return count;
    }

    @Override
    public int[] findThePrefixCommonArray(int[] A, int[] B) {
        int length = A.length;
        int[] res = new int[length];
        Set<Integer> setA = new HashSet<>();
        Set<Integer> setB = new HashSet<>();
        int count = 0;
        for(int i = 0; i < length; i++){
            if(!setA.contains(A[i])){
                setA.add(A[i]);
                if(setB.contains(A[i])){
                    count++;
                }
            }
            if(!setB.contains(B[i])){
                setB.add(B[i]);
                if(setA.contains(B[i])){
                    count++;
                }
            }
            res[i] = count;
        }
        return res;
    }

    @Override
    public String gcdOfStrings(String str1, String str2) {
        if(!(str1.concat(str2)).equals(str2.concat(str1))){
            return "";
        }
        int gcd = GCD(str1.length(), str2.length());

        return str1.substring(0,gcd);
    }

    @Override
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = Arrays.stream(candies).max().getAsInt();
        List<Boolean> result = new ArrayList<>();
        for(int candy : candies){
            if(candy + extraCandies >= max){
                result.add(true);
            }else{
                result.add(false);
            }
        }
        return result;
    }

    @Override
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int length = flowerbed.length;
        for(int i = 0; i < length; i++){
            int left = i == 0 ? 0 : flowerbed[i - 1];
            int right = i == length - 1 ? 0 : flowerbed[i + 1];
            if(left + right + flowerbed[i] == 0){
                --n;
                flowerbed[i] = 1;
            }
        }
        return n <= 0;
    }

    @Override
    public String reverseVowels(String s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            while(left < right && !isVowels(Character.toString(s.charAt(left)))){
                left++;
            }
            while(left < right && !isVowels(Character.toString(s.charAt(right)))){
                right--;
            }
            if(isVowels(Character.toString(s.charAt(left))) && isVowels(Character.toString(s.charAt(right)))){
                s = swapCharacters(s, left, right);
                left++;
                right--;
            }
        }
        return s;
    }

    @Override
    public String reverseWords(String s) {
        String[] newS = s.trim().split(" ");
        for(int i = 0; i < newS.length / 2; i++){
            String temp = newS[i];
            newS[i] = newS[newS.length - i - 1];
            newS[newS.length - i - 1] = temp;
        }
        return String.join(" ", newS);
    }

    @Override
    public String frequencySort(String s) {
        Map<Character, Integer> mapCount = new HashMap<>();
        for(char c : s.toCharArray()){
            mapCount.put(c, mapCount.getOrDefault(c, 0) + 1);
        }
        List<Character> list = new ArrayList<>(mapCount.keySet());

        list.sort((a,b) -> mapCount.get(b) - mapCount.get(a));

        StringBuilder sb = new StringBuilder();
        for(char c : list){
            for(int i = mapCount.get(c); i > 0; i--){
                sb.append(c);
            }
        }
        return sb.toString();
    }

    @Override
    public int jumpingOnClouds(int[] c, int k) {
        int energy = 100;
        Set<Integer> numbers = new HashSet<>();
        for(int i = 0; i < c.length; i = (i + k) % c.length){
            if(!numbers.contains(i)){
                numbers.add(i);
                energy -= 1;
                if(c[i] == 1){
                    energy -= 2;
                }
            }else{
                break;
            }
        }
        return energy;
    }

    @Override
    public int[][] highestPeak(int[][] isWater) {
            int m = isWater.length;
            int n = isWater[0].length;
            int[][] heights = new int[m][n];

            Queue<int[]> queue = new ArrayDeque<>();
            for(int i = 0; i < m; i++){
                for (int j = 0; j < n; j++){
                    heights[i][j] = isWater[i][j] - 1;

                        if(heights[i][j] == 0){
                        queue.offer(new int[]{i, j});
                    }
                }
            }

            int[] dirs = {-1, 0, 1, 0, -1};
            while (!queue.isEmpty()){
                int[] pos = queue.poll();
                int x = pos[0];
                int y = pos[1];
                for(int i = 0; i < 4; i++){
                    int newX = x + dirs[i];
                    int newY = y + dirs[i + 1];
                    if(newX >= 0 && newX < m && newY >= 0 && newY < n
                            && heights[newX][newY] == -1){
                        heights[newX][newY] = heights[x][y] + 1;

                        queue.offer(new int[]{newX, newY});
                    }
                }
            }
            return heights;
    }

    @Override
    public int countServers(int[][] grid) {
        Queue<int[]> queue = new ArrayDeque<>();
        int row = grid.length;
        int col = grid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if(grid[i][j] == 1){
                    queue.offer(new int[]{i, j});
                }
            }
        }
        int[] dirs = {-1, 0, 1, 0, -1};
        int count = 0;
        while (!queue.isEmpty()){
            int[] pos = queue.poll();
            for(int i = 0; i < 4; i++){
                int x = pos[0];
                int y = pos[1];
                int deltaX = dirs[i];
                int deltaY = dirs[i + 1];
                boolean isConnect = false;
                while(x + deltaX >= 0 && x + deltaX < row && y + deltaY >= 0 && y + deltaY < col){
                        x += deltaX;
                        y += deltaY;
                        if(grid[x][y] == 1){
                            isConnect = true;
                            break;
                        }
                }
                if(isConnect){
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    @Override
    public boolean check(int[] nums) {
        int countIncrease = 0;
        for(int i = 0 ; i < nums.length; i++){
            if(nums[i] > nums[(i+1) % nums.length]){
                countIncrease++;
            }
        }
        return countIncrease <= 1;
    }

    @Override
    public int maxAscendingSum(int[] nums) {
        int max = nums[0];
        int sum = max;
        for(int i = 1; i < nums.length; i++){
            if(nums[i] > nums[i - 1]){
                sum += nums[i];
            }else{
                sum = 0;
                sum += nums[i];
            }
            max = Math.max(max, sum);

        }
        return max;
    }

    @Override
    public boolean areAlmostEqual(String s1, String s2) {
        if(s1.equals(s2)){
            return true;
        }
        if(s1.length() != s2.length()){
            return false;
        }
        int mismatch = 0;
        char c1 = 0;
        char c2 = 0;
        for(int i = 0; i < s1.length(); i++){
            char currentC1 = s1.charAt(i);
            char currentC2 = s2.charAt(i);
            if(currentC1 != currentC2){
                mismatch++;
                if(mismatch > 2 || (mismatch == 2 && !(c1 == currentC2 && c2 == currentC1))){
                    return false;
                }
                c1 = currentC1;
                c2 = currentC2;
            }
        }
        return mismatch != 1;
    }

    @Override
    public void moveZeroes(int[] nums) {
        int index = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != 0){
                int temp = nums[i];
                nums[i] = nums[index];
                nums[index] = temp;
                index++;
            }
        }
    }

    @Override
    public boolean isSubsequence(String s, String t) {
        int lengthS = s.length();
        int lengthT = t.length();
        int indexS = 0;
        int indexT = 0;
        while(indexS < lengthS && indexT < lengthT){
            char charS = s.charAt(indexS);
            char charT = t.charAt(indexT);
            if(charS == charT){
                indexS++;
            }
            indexT++;
        }
        return indexS == lengthS;
    }

    @Override
    public double findMaxAverage(int[] nums, int k) {
       int currentSum = 0;
       for(int i = 0; i < k; i++){
           currentSum += nums[i];
       }
       int max = currentSum;
       for(int i = k; i < nums.length; i++){
           currentSum += (nums[i] - nums[i - k]);
           max = Math.max(max, currentSum);
       }
       return max * 1.0 / k;
    }

    @Override
    public int tupleSameProduct(int[] nums) {
        Map<Integer, Integer> countMap = new HashMap<>();
        for(int i = 0; i < nums.length - 1; i++){
            for(int j = i + 1; j < nums.length; j++){
                countMap.merge(nums[i] * nums[j],1,Integer::sum);
            }
        }
        int res = 0;
        for(int count : countMap.values()){
            res += count * (count - 1 ) / 2;
        }
        return res << 3;
    }

    @Override
    public int maxVowels(String s, int k) {
        int currentVowels = 0;
        for(int i = 0; i < k; i++){
            if(isVowels(s.charAt(i) + "")){
                currentVowels++;
            }
        }
        int maxVowel = currentVowels;
        for(int i = k; i < s.length(); i++){
            char currentChar = s.charAt(i);
            char firstChar = s.charAt(i - k);
            if(isVowels(firstChar + "")){
                currentVowels--;
            }
            if(isVowels(currentChar + "")){
                currentVowels++;
            }
            maxVowel = Math.max(maxVowel, currentVowels);
        }
        return maxVowel;
    }

    @Override
    public String clearDigits(String s) {
        StringBuilder sb = new StringBuilder();
        int[] dump = new int[s.length()];
        int index = 0;
        for(int i = 0 ; i < s.length(); i++){
            char current = s.charAt(i);
            if(Character.isDigit(current)){
                index = i;
                for(int j = index; j >= 0; j--){
                    char indexChar = s.charAt(j);
                    if(!Character.isDigit(indexChar) && dump[j] != -1){
                        dump[i] = -1;
                        dump[j] = -1;
                        break;
                    }
                }
            }
        }
        for(int i = 0 ; i < dump.length; i++){
            char c = s.charAt(i);
            if(dump[i] != -1){
                sb.append(c);
            }
        }
        return sb.toString();
    }

    @Override
    public int[] queryResults(int limit, int[][] queries) {
        Map<Integer, Integer> pairball = new HashMap<>();
        Map<Integer, Integer> countMap = new HashMap<>();
        int[] res = new int[queries.length];
        for(int i = 0; i < queries.length; i++){
            int ball = queries[i][0];
            int color = queries[i][1];

            countMap.merge(color, 1, Integer::sum);

            if(pairball.containsKey(ball)){
                int oldPair = pairball.get(ball);
                countMap.merge(oldPair, -1, Integer::sum);
                if(countMap.get(oldPair) == 0){
                    countMap.remove(oldPair);
                }
            }
            pairball.put(ball, color);
            res[i] = countMap.size();
        }
        return res;
    }

    @Override
    public String removeOccurrences(String s, String part) {
        while(s.contains(part)){
           s = s.replaceFirst(part, "");
        }
        return s;
    }

    @Override
    public long countBadPairs(int[] nums) {
        Map<Integer, Integer> sumMap = new HashMap<>();
        long goodPair = 0;
        long maxPair = (long) nums.length * (nums.length - 1) / 2;
        for(int i = 0; i < nums.length; i++){
            int firstPair = nums[i] - i;
            goodPair += sumMap.getOrDefault(firstPair, 0);
            sumMap.put(firstPair, sumMap.getOrDefault(firstPair, 0) + 1);
        }

        return  maxPair - goodPair;
    }

    @Override
    public int maximumSum(int[] nums) {
        Map<Integer, List<Integer>> sumMap = new HashMap<>();
        for (int num : nums) {
            int sum = calculateEachOfNumber(num);
            if (!sumMap.containsKey(sum)) {
                sumMap.put(sum, new ArrayList<>(List.of(num)));
            } else {
                sumMap.get(sum).add(num);
            }
        }
        int max = -1;
        for(Map.Entry<Integer, List<Integer>> entry : sumMap.entrySet()){
            List<Integer> value = entry.getValue();
            if(value.size() >= 2){
                value.sort((o1, o2) -> o2 -o1);
                int sum = value.get(0) + value.get(1);
                max = Math.max(max, sum);
            }
        }

        return max;
    }

    @Override
    public boolean isArraySpecial(int[] nums) {
        for(int i = 0; i < nums.length; i++){
            if(i + 1 >= nums.length){
                break;
            }
            int nextValue = nums[i + 1];
            int current = nums[i];
            if((nextValue % 2 == 0 && current % 2 == 0) ||
            (nextValue % 2 != 0 && current % 2 != 0)){
                return false;
            }
        }
        return true;
    }

    private int calculateEachOfNumber(int num){
        int res = 0;
        while(num != 0){
            int remain = num % 10;
            res += remain;
            num /= 10;
        }
        return res;
    }

    public String swapCharacters(String str, int index1, int index2) {
        char[] charArray = str.toCharArray();  // Convert String to char array

        // Swap characters at index1 and index2
        char temp = charArray[index1];
        charArray[index1] = charArray[index2];
        charArray[index2] = temp;

        // Convert char array back to String
        return new String(charArray);
    }
    private boolean isVowels(String s){
        String vowels = "aeiou";
        return vowels.contains(s.toLowerCase());
    }

    private int GCD(int a, int b){
        return b == 0 ? a : GCD(b, a % b);
    }
    private List<int[]> parseString(String stringToParse){
        List<int[]> res = new ArrayList<>();
        for(int i = 0; i < stringToParse.length(); i++){
            char c = stringToParse.charAt(i);
            if (c == 'L') {
                // 1 will be represent for L
                res.add(new int[]{1, i});
            }
            if(c == 'R'){
                res.add(new int[]{0, i});
            }
        }
        return res;
    }

    private int calculateNumber(int n){
        int sum = 0;
        while(n > 0){
            sum += (n % 10 )*( n % 10);
            n /= 10;
        }
        return sum;
    }

    private boolean match(char leftBracket, char rightBracket) {
        // Return true if pairs match, false otherwise
        return (leftBracket == '(' && rightBracket == ')') ||
                (leftBracket == '{' && rightBracket == '}') ||
                (leftBracket == '[' && rightBracket == ']');
    }
}
