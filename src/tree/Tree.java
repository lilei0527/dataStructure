package tree;
@SuppressWarnings("unused")
public interface Tree<K, V> {


    void insert(K k,V v);

    V search(K key);

    V delete(K key);

}
