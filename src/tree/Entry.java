package tree;

public class Entry<K, V> implements Comparable<Entry<K,V>> {
    /*存储索引关键字*/
    public K key;
    /*存储数据*/
    public V value;

    @Override
    public String toString() {
        return "Entry{" +
                "key=" + key +
                '}';
    }

    public Entry(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public void setKey(K key) {
        this.key = key;
    }

    public V getValue() {
        return value;
    }

    public void setValue(V value) {
        this.value = value;
    }

    @SuppressWarnings("unchecked")
    @Override
    public int compareTo(Entry<K,V> o) {
        Comparable<? super K> k = (Comparable<? super K>) key;
        return k.compareTo(o.getKey());
    }
}
