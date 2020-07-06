package tree.huffman;

import tree.Coded;

import java.io.*;
import java.util.*;

/**
 * @author lilei
 **/
public class DynamicHuffmanTree implements Coded {
    private DynamicNode NYT = new DynamicNode(0, 100, null);
    private Map<Byte, DynamicNode> nodeMap = new HashMap<>();
    private List<DynamicNode> allNode = new ArrayList<>();
    private DynamicNode root = NYT;

    private DynamicNode createNYT(byte b) {
        DynamicNode node = new DynamicNode(1, b, NYT.number - 1);
        nodeMap.put(b, node);
        DynamicNode dynamicNode = new DynamicNode(1, NYT, node, NYT.number, NYT.parent);
        if (NYT.number == 100) {
            root = dynamicNode;
        }
        if (NYT.parent != null)
            NYT.parent.left = dynamicNode;
        allNode.add(node);
        allNode.add(dynamicNode);
        node.parent = dynamicNode;
        NYT.parent = dynamicNode;
        NYT.number = NYT.number - 2;
        return dynamicNode;
    }

    private StringBuilder addNode(byte b) {
        StringBuilder stringBuilder = new StringBuilder();
        if (!nodeMap.containsKey(b)) {
            DynamicNode nyt = createNYT(b);
            String code = getCode(nyt);
            stringBuilder.append(code);
            stringBuilder.append(toString(b));
            swap(nyt.parent);
            addWeight(nyt.parent);
        } else {
            DynamicNode node = nodeMap.get(b);
            String code = getCode(node);
            stringBuilder.append(code);
            swap(node);
            addWeight(node);
        }
        return stringBuilder;
    }


    private void addWeight(DynamicNode node) {
        while (node != null) {
            node.weight++;
            node = node.parent;
        }
    }

    private byte[] toArray(File file) {
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(file);
            byte[] bytes = new byte[(int) file.length()];
            inputStream.read(bytes);
            return bytes;
        } catch (IOException e) {
            return null;
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private StringBuilder toString(byte b) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < 8; i++) {
            if ((b & (1 << i)) == 1 << i) {
                stringBuilder.append("1");
            } else {
                stringBuilder.append("0");
            }
        }
        return stringBuilder;
    }

    private void swap(DynamicNode node) {
        while (node != null) {
            DynamicNode maxNumberNode = getMaxNumberNode(node.weight);
            if (maxNumberNode != node && maxNumberNode != node.parent) {
                if (node.parent.left == NYT) {
                    swap(node.parent, maxNumberNode);
                } else {
                    swap(node, maxNumberNode);
                }
            }
//            node.weight++;
            node = node.parent;
        }
    }

    private String getCode(DynamicNode node) {
        StringBuilder stringBuilder = new StringBuilder();
        while (node.parent != null) {
            if (node.parent.left == node) {
                stringBuilder.append("0");
            } else {
                stringBuilder.append("1");
            }
            node = node.parent;
        }
        return stringBuilder.reverse().toString();
    }

//    private byte[] coverCodeString(StringBuilder code) {
//        byte[] bytes = new byte[code.length()];
//        int index = 0;
//        for (int i = 0; i < code.length(); i = i + 8) {
//            bytes[index++] = (byte) Integer.parseInt(code.substring(i, i + 8), 2);
//        }
//        return bytes;
//    }

    private void swap(DynamicNode org, DynamicNode des) {
        int temp = org.number;
        org.number = des.number;
        des.number = temp;


        DynamicNode orgParent = org.parent;
        DynamicNode desParent = des.parent;

        if (orgParent == desParent) {
            if (orgParent.left == org) {
                orgParent.left = des;
                orgParent.right = org;
            } else {
                orgParent.right = des;
                orgParent.left = org;
            }
        } else {
            des.parent = orgParent;
            org.parent = desParent;

            if (orgParent.left == org) {
                orgParent.left = des;
            } else {
                orgParent.right = des;
            }

            if (desParent.left == des) {
                desParent.left = org;
            } else {
                desParent.right = org;
            }
        }

    }

    //查找权重相同节点中节点编号最大的节点
    private DynamicNode getMaxNumberNode(int weight) {
        DynamicNode max = null;
        for (DynamicNode node : allNode) {
            if (node.weight == weight) {
                if (max == null) {
                    max = node;
                } else {
                    max = node.number > max.number ? node : max;
                }
            }
        }
        return max;
    }


    @Override
    public byte[] encode(byte[] bytes) {
        byte[] newBytes = new byte[bytes.length];
        int index = 0;
        for (byte b : bytes) {
            StringBuilder stringBuilder = addNode(b);
            for (int i = 0; i < stringBuilder.length(); i++) {
                if (stringBuilder.charAt(i) == '1') {
                    if (index >= newBytes.length * 8) {
                        newBytes = resize(newBytes, 2);
                    }
                    newBytes[index / 8] |= (1 << (index % 8));
                }
                index++;
            }
        }

        int remain = 8 - (index - 1) % 8;//剩余补0的位数
        String code = getCode(NYT);

        for (int i = 0; i < remain; i++) {
            if (code.charAt(i % code.length()) == '1') {
                if (index >= newBytes.length * 8) {
                    newBytes = resize(newBytes, 2);
                }
                newBytes[index / 8] |= (1 << (index % 8));
            }
            index++;
        }

        int size = getSize(index - 1);
        byte[] returnBytes = new byte[size];
        System.arraycopy(newBytes, 0, returnBytes, 0, size);
        return returnBytes;
    }

    private byte[] resize(byte[] bytes, int times) {
        byte[] resizeBytes = new byte[bytes.length * times];
        System.arraycopy(bytes, 0, resizeBytes, 0, bytes.length);
        return resizeBytes;
    }

    public byte[] encode(String s) {
        return encode(s.getBytes());
    }

    public byte[] encode(File file) {
        byte[] bytes = toArray(file);
        if (bytes == null) {
            return null;
        }
        return encode(bytes);
    }

    @Override
    public byte[] decode(byte[] bytes) {
        byte[] newBytes = new byte[bytes.length];
        StringBuilder stringBuilder = new StringBuilder();
        int index = 0;
        int newIndex = 0;
        while (index < bytes.length * 8) {
            DynamicNode node = getNode(stringBuilder);
            if (node != null) {
                //NYT
                if (node.weight == 0) {
                    //如果不够一个字节
                    if (index > (bytes.length - 1) * 8) {
                        newIndex += bytes.length * 8 - index;
                        break;
                    }
                    //读入一个字节
                    byte b = 0x00;
                    for (int i = 0; i < 8; i++) {
                        //bytes的最新位为1
                        if ((bytes[index / 8] & (1 << (index % 8))) == 1 << (index % 8)) {
                            b |= 1 << i;
                            if (index >= newBytes.length * 8) {
                                newBytes = resize(newBytes, 2);
                            }
                            newBytes[newIndex / 8] |= 1 << (newIndex % 8);
                        }
//                        newBytes[newIndex / 8] |= (bytes[index / 8] & (1 << (index % 8)));
                        index++;
                        newIndex++;
                    }
                    addNode(b);
                } else {
                    //leaf
                    for (int i = 0; i < 8; i++) {
                        if (1 << i == (node.k & (1 << i))) {
                            if (index >= newBytes.length * 8) {
                                newBytes = resize(newBytes, 2);
                            }
                            newBytes[newIndex / 8] |= 1 << (newIndex % 8);
                        }
                        newIndex++;
                    }
                    addNode(node.k);
                }
                stringBuilder.delete(0, stringBuilder.length());
            }

            if ((bytes[index / 8] & (1 << (index % 8))) == 1 << (index % 8)) {
                stringBuilder.append("1");
            } else {
                stringBuilder.append("0");
            }
            index++;
        }
        int size = getSize(newIndex - 1);
        byte[] reBytes = new byte[size];
        System.arraycopy(newBytes, 0, reBytes, 0, size);
        return reBytes;
    }

    private int getSize(int i) {
        return i % 8 == 0 ? i / 8 : i / 8 + 1;
    }


    private DynamicNode getNode(StringBuilder code) {
        DynamicNode node = root;
        if (code.length() == 0) {
            return node;
        }
        for (int i = 0; i < code.length(); i++) {
            if (code.charAt(i) == '0') {
                node = (DynamicNode) node.left;
            } else {
                node = (DynamicNode) node.right;
            }
            if (node == null) {
                throw new RuntimeException();
            }
            if (node.left == null && node.right == null) {
                return node;
            }
        }
        return null;
    }

    public static void main(String[] args) throws IOException {
        DynamicHuffmanTree tree = new DynamicHuffmanTree();
        String s = "abcdefghijklmnopqrstuvwxyz";
        byte[] encode = tree.encode(s);
        System.out.println(Arrays.toString(s.getBytes()));
        System.out.println(Arrays.toString(encode));

        DynamicHuffmanTree tree1 = new DynamicHuffmanTree();
        byte[] decode = tree1.decode(encode);
        System.out.println(Arrays.toString(decode));

//        DynamicHuffmanTree tree2 = new DynamicHuffmanTree();
//        byte[] encode1 = tree2.encode(new File("C:\\Users\\lilei\\Desktop\\1.jpg"));
//        OutputStream outputStream = new FileOutputStream(new File("C:\\Users\\lilei\\Desktop\\1.zip"));
//        outputStream.write(encode1);
//
//        DynamicHuffmanTree tree3 = new DynamicHuffmanTree();
//        byte[] decode = tree3.decode(encode1);
//        OutputStream outputStream1 = new FileOutputStream(new File("C:\\Users\\lilei\\Desktop\\2.jpg"));
//        outputStream1.write(decode);
    }
}