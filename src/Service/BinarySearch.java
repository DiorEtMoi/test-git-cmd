package Service;

public class BinarySearch implements ISearch{
    @Override
    public boolean search(int[] a, int target) {
        int low = 0;
        int height = a.length - 1;
        while(low <= height){
            int mid = height / 2;
            if(a[mid] == target){
                return true;
            }else if(a[mid] > target){
                height = mid - 1;
            }else if(a[mid] < target){
                low = mid + 1;
            }
        }
        return false;
    }
}