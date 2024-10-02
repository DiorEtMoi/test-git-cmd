import services.BinarySearch;
import services.ISearch;

// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
            ISearch search = new BinarySearch();
            int a[] = {1,2,3,4,5,6,7,8,9,10};
            boolean found = search.search(a, 11);
            System.out.println("Found : " + found);
        }
    }
