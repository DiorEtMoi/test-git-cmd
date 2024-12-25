package services;

public class LinkedListNode {
    public LinkedListNode next;
    public String value;

    public LinkedListNode(String value) {
        this.value = value;
    }
    public LinkedListNode(String value, LinkedListNode next) {
        this.value = value;
        this.next = next;
    }
}
