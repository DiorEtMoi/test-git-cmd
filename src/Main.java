import services.ILeetCode;
import services.LeetCode;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    public static void main(String[] args){
        ILeetCode leetcode = new LeetCode();
        int[] a = {368,369,307,304,384,138,90,279,35,396,114,328,251,364,300,191,438,467,183};
        System.out.println(leetcode.maximumSum(a));
    }

}