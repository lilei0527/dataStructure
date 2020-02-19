package tree;

public class Entry implements Comparable<Entry>{
    /*存储索引关键字*/
    private int key;
    /*存储数据*/
    private Object value;

    @Override
    public String toString() {
        return "Entry{" +
                "key=" + key +
                '}';
    }

    @Override
    public int compareTo(Entry o) {
        //根据key的值升序排列
        return this.key - o.key;
    }

    public int getKey() {
        return key;
    }

    public void setKey(int key) {
        this.key = key;
    }

    public Object getValue() {
        return value;
    }

    public void setValue(Object value) {
        this.value = value;
    }

     public Entry(int key, Object value) {
        this.key = key;
        this.value = value;
    }
}
