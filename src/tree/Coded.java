package tree;

//可编码
public interface Coded<K> {
    byte[] encode(byte[]k);
    byte[]decode(byte[]bytes);

}
