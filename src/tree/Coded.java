package tree;

//可编码
public interface Coded<K> {
    void listCode();

    byte[] code(K[] key);

    byte[] getCode(K k);
}
