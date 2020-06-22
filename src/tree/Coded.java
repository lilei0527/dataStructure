package tree;

//可编码
public interface Coded<K> {
    byte[] encode(K[]k);
    byte[]decode(byte[]bytes);
    String getCode(K k);
}
