
import services.ILeetCode;
import services.LeetCode;


// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    public static void main(String[] args){
        ILeetCode leet = new LeetCode();
        int[] nums = {1,2,4,6};
        System.out.println(leet.productExceptSelf(nums));
    }
}