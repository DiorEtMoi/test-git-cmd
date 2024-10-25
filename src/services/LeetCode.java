package services;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

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

    @Override
    public boolean isValid(String s) {
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
    private boolean match(char leftBracket, char rightBracket) {
        // Return true if pairs match, false otherwise
        return (leftBracket == '(' && rightBracket == ')') ||
                (leftBracket == '{' && rightBracket == '}') ||
                (leftBracket == '[' && rightBracket == ']');
    }
}
