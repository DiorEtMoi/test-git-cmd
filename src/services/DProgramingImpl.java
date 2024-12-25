package services;

public class DProgramingImpl implements DPrograming {
    @Override
    public int findingMinimunCoins(int total, int[] weights) {
       int[] storeCoins = new int[total + 1];
       for(int p = 1; p <= total; p++) {
           for(int w : weights){
               if(w <= p){
                   storeCoins[p] = Math.min(storeCoins[p - 1], storeCoins[p - w]) + 1;
               }
           }
       }
       return storeCoins[total];
    }
}
