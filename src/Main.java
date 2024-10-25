import services.ILeetCode;
import services.LeetCode;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        ILeetCode iLeetCode = new LeetCode();
        String s = "()";
        System.out.println(iLeetCode.isValid(s));
    }
}