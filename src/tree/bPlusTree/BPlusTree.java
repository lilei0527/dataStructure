package tree.bPlusTree;

import tree.Entry;
import tree.Tree;

import java.util.*;

@SuppressWarnings("unused")
public class BPlusTree<K extends Comparable<K>, V> implements Tree<K, V> {
    private static final String PREN_ODE = "PREN_ODE";
    private static final String NEXT_NODE = "NEXT_NODE";
    //B+树的阶数
    private int rank;
    //根节点
    private Node<K, V> root;
    //头结点
    private Node<K, V> head;

    public BPlusTree(int rank) {
        this.rank = rank;
    }

    public Node<K, V> getRoot() {
        return root;
    }

    public void insert(K k, V v) {
        Entry<K, V> entry = new Entry<>(k, v);
        List<Entry<K, V>> keyAndValues1 = new ArrayList<>();
        //插入第一个节点
        if (head == null) {
            keyAndValues1.add(entry);
            head = new Node<>(null, keyAndValues1, null, null, null);
            root = new Node<>(null, keyAndValues1, null, null, null);
        } else {
            Node<K, V> node = head;
            //遍历链表，找到插入键值对对应的节点
            while (node != null) {
                List<Entry<K, V>> entries = node.getEntry();
                int exitFlag = 0;
                //如果插入的键的值和当前节点键值对集合中的某个键的值相等，则直接替换value
                for (Entry<K, V> KV : entries) {
                    if (KV.getKey() == entry.getKey()) {
                        KV.setValue(entry.getValue());
                        exitFlag = 1;
                        break;
                    }
                }
                //如果插入的键已经有了，则退出循环
                if (exitFlag == 1) {
                    break;
                }
                //如果当前节点是最后一个节点或者要插入的键值对的键的值小于下一个节点的键的最小值，则直接插入当前节点

                if (node.getNextNode() == null || node.getNextNode().getEntry().get(0).getKey().compareTo(entry.getKey()) >= 0) {
                    splitNode(node, entry);
                    break;
                }
                //移动指针
                node = node.getNextNode();
            }
        }
    }

    @Override
    public V search(K key) {
        Node<K, V> search = search(key, root);
        return getKeyAndValue(search.getEntry(), key);
    }


    //判断是否需要拆分节点
    private void splitNode(Node<K, V> node, Entry<K, V> entry) {
        List<Entry<K, V>> entries = node.getEntry();

        if (entries.size() == rank - 1) {
            //先插入待添加的节点
            entries.add(entry);
            Collections.sort(entries);
            //取出当前节点的键值对集合
            //取出原来的key-value集合中间位置的下标
            int mid = entries.size() / 2;
            //取出原来的key-value集合中间位置的键
            K midKey = entries.get(mid).getKey();
            //构造一个新的键值对，不是叶子节点的节点不存储value的信息
            Entry<K, V> midEntry = new Entry<>(midKey, null);
            //将中间位置左边的键值对封装成集合对象
            List<Entry<K, V>> leftEntries = new ArrayList<>();
            for (int i = 0; i < mid; i++) {
                leftEntries.add(entries.get(i));
            }
            //将中间位置右边边的键值对封装成集合对象
            List<Entry<K, V>> rightEntries = new ArrayList<>();
            //如果是叶子节点则在原节点中保留上移的key-value，否则原节点删除上移的key-value
            int k;
            if (node.isLeaf()) {
                k = mid;
            } else {
                k = mid + 1;
            }
            for (int i = k; i < rank; i++) {
                rightEntries.add(entries.get(i));
            }
            //对左右两边的元素重排序
            Collections.sort(leftEntries);
            Collections.sort(rightEntries);
            //以mid为界限将当前节点分列成两个节点，维护前指针和后指针
            Node<K, V> rightNode;
            Node<K, V> leftNode;
//            if (node.isLeaf()) {
            //如果是叶子节点维护前后指针
            rightNode = new Node<>(null, rightEntries, node.getNextNode(), null, node.getParentNode());
            leftNode = new Node<>(null, leftEntries, rightNode, node.getPreviousNode(), node.getParentNode());
            rightNode.setPreviousNode(leftNode);
//            } else {
//                //如果不是叶子不维护前后指针
//                rightNode = new Node(null, rightKeyAndValues, null, null, node.getParentNode());
//                leftNode = new Node(null, leftKeyAndValues, null, null, node.getParentNode());
//            }
            //如果当前分裂的节点有孩子节点,设置分裂后节点和孩子节点的关系
            if (node.getNodes() != null) {
                //取得所有地孩子节点
                List<Node<K, V>> nodes = node.getNodes();
                List<Node<K, V>> leftNodes = new ArrayList<>();
                List<Node<K, V>> rightNodes = new ArrayList<>();
                for (Node<K, V> childNode : nodes) {
                    //取得当前孩子节点的最大键值
                    K max = childNode.getEntry().get(childNode.getEntry().size() - 1).getKey();
                    if (max.compareTo(midEntry.getKey()) < 0) {
                        //小于mid处的键的数是左节点的子节点
                        leftNodes.add(childNode);
                        childNode.setParentNode(leftNode);
                    } else {
                        //大于mid处的键的数是右节点的子节点
                        rightNodes.add(childNode);
                        childNode.setParentNode(rightNode);
                    }
                }
                leftNode.setNodes(leftNodes);
                rightNode.setNodes(rightNodes);
            }

            //当前节点的前节点
            Node<K, V> preNode = node.getPreviousNode();
            //分裂节点后将分裂节点的前节点的后节点设置为左节点
            if (preNode != null) {
                preNode.setNextNode(leftNode);
            }

            //当前节点的后节点
            Node<K, V> nextNode = node.getNextNode();
            //分裂节点后将分裂节点的后节点的前节点设置为右节点
            if (nextNode != null) {
                nextNode.setPreviousNode(rightNode);
            }

            //如果由头结点分裂，则分裂后左边的节点为头节点
            if (node == head) {
                head = leftNode;
            }

            //父节点的子节点
            List<Node<K, V>> childNodes = new ArrayList<>();
            childNodes.add(rightNode);
            childNodes.add(leftNode);
            //分裂
            //当前节点无父节点
            if (node.getParentNode() == null) {
                //父节点的键值对
                List<Entry<K, V>> parentEntries = new ArrayList<>();
                parentEntries.add(midEntry);
                //构造父节点
                Node<K, V> parentNode = new Node<>(childNodes, parentEntries, null, null, null);
                //将子节点与父节点关联
                rightNode.setParentNode(parentNode);
                leftNode.setParentNode(parentNode);
                //当前节点为根节点
                root = parentNode;
            } else {
                Node<K, V> parentNode = node.getParentNode();
                //将原来的孩子节点（除了被拆分的节点）和新的孩子节点（左孩子和右孩子）合并之后与父节点关联
                childNodes.addAll(parentNode.getNodes());
                //移除正在被拆分的节点
                childNodes.remove(node);
                //将子节点与父节点关联
                parentNode.setNodes(childNodes);
                rightNode.setParentNode(parentNode);
                leftNode.setParentNode(parentNode);
                if (parentNode.getParentNode() == null) {
                    root = parentNode;
                }
                //当前节点有父节点,递归调用拆分的方法,将父节点拆分
                splitNode(parentNode, midEntry);
            }
        } else {
            entries.add(entry);
            //排序
            Collections.sort(entries);
        }
    }


    //打印B+树
    public void print() {
        print(root);
    }

    //打印B+树
    void print(Node<K, V> root) {
        if (root == this.root) {
            //打印根节点内的元素
            printNode(root);
            System.out.println();
        }
        if (root == null) {
            return;
        }

        //打印子节点的元素
        if (root.getNodes() != null) {
            //找到最左边的节点
            Node<K, V> leftNode = null;
            Node<K, V> tmpNode = null;
            List<Node<K, V>> childNodes = root.getNodes();
            for (Node<K, V> node : childNodes) {
                if (node.getPreviousNode() == null) {
                    leftNode = node;
                    tmpNode = node;
                }
            }

            while (leftNode != null) {
                //从最左边的节点向右打印
                printNode(leftNode);
                System.out.print("|");
                leftNode = leftNode.getNextNode();
            }
            System.out.println();
            print(tmpNode);
        }
    }

    //打印一个节点内的元素
    private void printNode(Node<K, V> node) {
        List<Entry<K, V>> entries = node.getEntry();
        for (int i = 0; i < entries.size(); i++) {
            if (i != (entries.size() - 1)) {
                System.out.print(entries.get(i).getKey() + ",");
            } else {
                System.out.print(entries.get(i).getKey());
            }
        }
    }

    public Node<K, V> search(K key, Node<K, V> node) {

        //如果是叶子节点则直接取值
        if (node.isLeaf()) {
            List<Entry<K, V>> entries = node.getEntry();
            for (Entry<K, V> entry : entries) {
                if (entry.getKey() == key) {
                    return node;
                }
            }
            return null;
        }


        List<Node<K, V>> nodes = node.getNodes();
        //如果寻找的key小于节点的键的最小值
        K minKey = node.getEntry().get(0).getKey();
        if (key.compareTo(minKey) < 0) {
            for (Node<K, V> n : nodes) {
                List<Entry<K, V>> entries = n.getEntry();
                //找到子节点集合中最大键小于父节点最小键节点
                if (minKey.compareTo(entries.get(entries.size() - 1).getKey()) > 0) {
                    return search(key, n);
                }
            }
        }
        //如果寻找的key大于节点的键的最大值
        K maxKey = getMaxKeyInNode(node);
        if (key.compareTo(maxKey) >= 0) {
            if (key.compareTo(maxKey) >= 0) {
                for (Node<K, V> n : nodes) {
                    List<Entry<K, V>> entries = n.getEntry();
                    //找到子节点集合中最小键大于等于父节点最小大键节点
                    if (maxKey.compareTo(entries.get(0).getKey()) <= 0) {
                        return search(key, n);
                    }
                }
            }
        }

        //如果寻找的key在最大值和最小值之间，首先定位到最窄的区间
        K min = getLeftBoundOfKey(node, key);
        K max = getRightBoundOfKey(node, key);


        //去所有的子节点中找键的范围在min和max之间的节点
        for (Node<K, V> n : nodes) {
            List<Entry<K, V>> kvs = n.getEntry();
            //找到子节点集合中键的范围在min和max之间的节点
            if (min.compareTo(kvs.get(0).getKey()) <= 0 && max.compareTo(kvs.get(kvs.size() - 1).getKey()) > 0) {
                return search(key, n);
            }
        }
        return null;
    }


    public V delete(K key) {
        System.out.println("delete:" + key);
        System.out.println();

        //首先找到要删除的key所在的节点
        Node<K, V> deleteNode = search(key, root);
        //如果没找到则删除失败
        if (deleteNode == null) {
            return null;
        }

        if (deleteNode == root) {
            return delKeyAndValue(root.getEntry(), key);
        }

        if (deleteNode == head && isNeedMerge(head)) {
            head = head.getNextNode();
        }

        if (merge(deleteNode, key)) {
            return getKeyAndValue(deleteNode.getEntry(), key);
        } else {
            return null;
        }

    }


    //平衡当前节点和前节点或者后节点的数量，使两者的数量都满足条件
    private boolean balanceNode(Node<K, V> node, Node<K, V> brotherNode, String nodeType) {
        if (brotherNode == null) {
            return false;
        }
        List<Entry<K, V>> delEntries = node.getEntry();
        if (isMoreElement(brotherNode)) {
            List<Entry<K, V>> brotherEntries = brotherNode.getEntry();
            int brotherSize = brotherEntries.size();
            //兄弟节点删除挪走的键值对
            Entry<K, V> entry = null;
            Entry<K, V> entry1;
            switch (nodeType) {
                case PREN_ODE:
                    entry = brotherEntries.remove(brotherSize - 1);
                    entry1 = getKeyAndValueInMinAndMax(node.getParentNode(), entry.getKey(), getMinKeyInNode(node));
                    entry1.setKey(entry.getKey());
                    break;
                case NEXT_NODE:
                    entry = brotherEntries.remove(0);
                    entry1 = getKeyAndValueInMinAndMax(node.getParentNode(), getMaxKeyInNode(node), entry.getKey());
                    entry1.setKey(brotherEntries.get(0).getKey());
                    break;
            }
            //当前节点添加从前一个节点得来的键值对
            delEntries.add(entry);

            //对键值对重排序
            Collections.sort(delEntries);
            return true;
        }
        return false;
    }

    public boolean merge(Node<K, V> node, K key) {
        List<Entry<K, V>> delEntries = node.getEntry();
        //首先删除该key-value
        delKeyAndValue(delEntries, key);
        //如果要删除的节点的键值对的数目小于节点最大键值对数目*填充因子
        if (isNeedMerge(node)) {
            boolean isBalance;
            //如果左节点有富余的键值对，则取一个到当前节点
            Node<K, V> preNode = getPreviousNode(node);
            isBalance = balanceNode(node, preNode, PREN_ODE);
            //如果此时已经平衡，则已经删除成功
            if (isBalance) return true;

            //如果右兄弟节点有富余的键值对，则取一个到当前节点
            Node<K, V> nextNode = getNextNode(node);
            isBalance = balanceNode(node, nextNode, NEXT_NODE);

            return isBalance || mergeNode(node, key);
        } else {
            return true;
        }
    }

    //合并节点
    //key 待删除的key
    private boolean mergeNode(Node<K, V> node, K key) {
        if (node.isRoot()) {
            return false;
        }
        Node<K, V> preNode;
        Node<K, V> nextNode;
        Node<K, V> parentNode = node.getParentNode();
        List<Node<K, V>> childNodes = parentNode.getNodes();
        List<Node<K, V>> childNodes1 = node.getNodes();
        List<Entry<K, V>> parentEntry = parentNode.getEntry();
        List<Entry<K, V>> entries = node.getEntry();

        if (node.isLeaf()) {
            if (parentEntry.size() == 1 && parentNode != root) {
                return true;
            }
            preNode = getPreviousNode(node);
            nextNode = getNextNode(node);
            if (preNode != null) {
                List<Entry<K, V>> preEntries = preNode.getEntry();
                entries.addAll(preEntries);
                if (preNode.isHead()) {
                    head = node;
                    node.setPreviousNode(null);
                } else {
                    preNode.getPreviousNode().setNextNode(node);
                    node.setPreviousNode(preNode.getPreviousNode());
                }
                //将合并后节点的后节点设置为当前节点的后节点
                preNode.setNextNode(node.getNextNode());
                Entry<K, V> entry = getKeyAndValueInMinAndMax(parentNode, getMinKeyInNode(preNode), key);
                delKeyAndValue(parentEntry, entry.getKey());
                if (parentEntry.isEmpty()) {
                    root = node;
                } else {
                    //删除当前节点
                    childNodes.remove(preNode);
                }
                Collections.sort(entries);
                merge(parentNode, key);
                return true;
            }

            if (nextNode != null) {
                List<Entry<K, V>> nextEntries = nextNode.getEntry();
                entries.addAll(nextEntries);
                if (nextNode.isTail()) {
                    node.setPreviousNode(null);
                } else {
                    nextNode.getNextNode().setPreviousNode(node);
                    node.setNextNode(nextNode.getNextNode());
                }

                Entry<K, V> entry = getKeyAndValueInMinAndMax(parentNode, key, getMinKeyInNode(nextNode));
                delKeyAndValue(parentEntry, entry.getKey());
                if (parentEntry.isEmpty()) {
                    root = node;
                    node.setParentNode(null);
                } else {
                    //删除当前节点
                    childNodes.remove(nextNode);
                }
                Collections.sort(entries);
                merge(parentNode, key);
                return true;
            }
            //前节点和后节点都等于null那么是root节点
        } else {
            preNode = getPreviousNode(node);
            nextNode = getNextNode(node);
            if (preNode != null) {
                //将前一个节点和当前节点还有父节点中的相应Key-value合并
                List<Entry<K, V>> preEntries = preNode.getEntry();
                preEntries.addAll(entries);
                K min = getMaxKeyInNode(preNode);
                K max = getMinKeyInNode(node);
                //父节点中移除这个key-value
                Entry<K, V> entry = getKeyAndValueInMinAndMax(parentNode, min, max);
                parentEntry.remove(entry);
                if (parentEntry.isEmpty()) {
                    root = preNode;
                    node.setParentNode(null);
                    preNode.setParentNode(null);
                } else {
                    childNodes.remove(node);
                }
                assert nextNode != null;
                preNode.setNextNode(nextNode.getNextNode());
                //前节点加上一个当前节点的所有子节点中最小key的key-value
                Entry<K, V> minEntry = getMinKeyAndValueInChildNode(node);
                assert minEntry != null;
                Entry<K, V> entry1 = new Entry<>(minEntry.getKey(), minEntry.getValue());
                preEntries.add(entry1);
                List<Node<K, V>> preChildNodes = preNode.getNodes();
                preChildNodes.addAll(node.getNodes());
                //将当前节点的孩子节点的父节点设为当前节点的后节点
                for (Node<K, V> node1 : childNodes1) {
                    node1.setParentNode(preNode);
                }
                Collections.sort(preEntries);
                merge(parentNode, key);
                return true;
            }

            if (nextNode != null) {
                //将后一个节点和当前节点还有父节点中的相应Key-value合并
                List<Entry<K, V>> nextEntries = nextNode.getEntry();
                nextEntries.addAll(entries);

                K min = getMaxKeyInNode(node);
                K max = getMinKeyInNode(nextNode);
                //父节点中移除这个key-value
                Entry<K, V> entry = getKeyAndValueInMinAndMax(parentNode, min, max);
                parentEntry.remove(entry);
                childNodes.remove(node);
                if (parentEntry.isEmpty()) {
                    root = nextNode;
                    nextNode.setParentNode(null);
                } else {
                    childNodes.remove(node);
                }
                nextNode.setPreviousNode(node.getPreviousNode());
                //后节点加上一个当后节点的所有子节点中最小key的key-value
                Entry<K, V> minEntry = getMinKeyAndValueInChildNode(nextNode);
                assert minEntry != null;
                Entry<K, V> entry1 = new Entry<>(minEntry.getKey(), minEntry.getValue());
                nextEntries.add(entry1);
                List<Node<K, V>> nextChildNodes = nextNode.getNodes();
                nextChildNodes.addAll(node.getNodes());
                //将当前节点的孩子节点的父节点设为当前节点的后节点
                for (Node<K, V> node1 : childNodes1) {
                    node1.setParentNode(nextNode);
                }
                Collections.sort(nextEntries);
                merge(parentNode, key);
                return true;
            }
        }
        return false;
    }

    //得到当前节点的前节点
    @SuppressWarnings({"unchecked", "rawtypes"})
    private Node<K, V> getPreviousNode(Node<K, V> node) {
        if (node.isRoot()) {
            return null;
        }
        Node<K, V> parentNode = node.getParentNode();
        //得到兄弟节点
        List<Node<K, V>> nodes = parentNode.getNodes();
        List<Entry<K, V>> entries = new ArrayList<>();

        for (Node<K, V> n : nodes) {
            List<Entry<K, V>> list = n.getEntry();
            K maxKeyAndValue = list.get(list.size() - 1).getKey();
            if (maxKeyAndValue.compareTo(getMinKeyInNode(node)) < 0) {
                entries.add(new Entry(maxKeyAndValue, n));
            }
        }
        Collections.sort(entries);
        if (entries.isEmpty()) {
            return null;
        }
        return (Node) entries.get(entries.size() - 1).getValue();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    //得到当前节点的后节点
    private Node<K, V> getNextNode(Node<K, V> node) {
        if (node.isRoot()) {
            return null;
        }
        Node<K, V> parentNode = node.getParentNode();

        //得到兄弟节点
        List<Node<K, V>> nodes = parentNode.getNodes();
        List<Entry<K, V>> entries = new ArrayList<>();
        for (Node<K, V> n : nodes) {
            List<Entry<K, V>> list = n.getEntry();
            K minKey = list.get(0).getKey();
            if (minKey.compareTo(getMaxKeyInNode(node)) > 0) {
                entries.add(new Entry(minKey, n));
            }
        }
        Collections.sort(entries);
        if (entries.isEmpty()) {
            return null;
        }
        return (Node) entries.get(0).getValue();
    }


    private K getMinKeyInNode(Node<K, V> node) {
        List<Entry<K, V>> entries = node.getEntry();
        return entries.get(0).getKey();
    }

    private K getMaxKeyInNode(Node<K, V> node) {
        List<Entry<K, V>> entries = node.getEntry();
        return entries.get(entries.size() - 1).getKey();
    }


    private K getLeftBoundOfKey(Node<K, V> node, K key) {
        K left = null;
        List<Entry<K, V>> entries = node.getEntry();
        for (int i = 0; i < entries.size(); i++) {
            if (key.compareTo(entries.get(i).getKey()) >= 0 && key.compareTo(entries.get(i + 1).getKey()) < 0) {
                left = entries.get(i).getKey();
                break;
            }
        }
        return left;
    }

    private K getRightBoundOfKey(Node<K, V> node, K key) {
        K right = null;
        List<Entry<K, V>> entries = node.getEntry();
        for (int i = 0; i < entries.size(); i++) {
            if (key.compareTo(entries.get(i).getKey()) >= 0 && key.compareTo(entries.get(i + 1).getKey()) < 0) {
                right = entries.get(i + 1).getKey();
                break;
            }
        }
        return right;
    }


    private V delKeyAndValue(List<Entry<K, V>> entries, K key) {
        for (Entry<K, V> entry : entries) {
            if (key.compareTo(entry.getKey()) == 0) {
                entries.remove(entry);
                return entry.value;
            }
        }
        return null;
    }

    private V getKeyAndValue(List<Entry<K, V>> entries, K key) {
        for (Entry<K, V> entry : entries) {
            if (entry.getKey().equals(key)) {
                return entry.value;
            }
        }
        return null;
    }

    //找到node的键值对中在min和max中的键值对
    @SuppressWarnings("all")
    private <K extends Comparable<? super K>> Entry<K, V> getKeyAndValueInMinAndMax(Node<K, V> node, K min, K max) {
        if (node == null) {
            return null;
        }
        List<Entry<K, V>> entries = node.getEntry();
        Entry<K, V> entry = null;
        for (Entry<K, V> k : entries) {
            if (min.compareTo(k.getKey()) < 0 && max.compareTo(k.getKey()) >= 0) {
                entry = k;
                break;
            }
        }
        return entry;
    }

//    private KeyAndValue getMaxKeyAndValueInChildNode(Node node) {
//        if (node.getNodes() == null || node.getNodes().isEmpty()) {
//            return null;
//        }
//        List<KeyAndValue> sortKeyAndValues = new ArrayList<>();
//        List<Node> childNodes = node.getNodes();
//        for (Node childNode : childNodes) {
//            List<KeyAndValue> keyAndValues = childNode.getKeyAndValue();
//            KeyAndValue maxKeyAndValue = keyAndValues.get(keyAndValues.size() - 1);
//            sortKeyAndValues.add(maxKeyAndValue);
//        }
//        Collections.sort(sortKeyAndValues);
//        return sortKeyAndValues.get(sortKeyAndValues.size() - 1);
//    }

    private Entry<K, V> getMinKeyAndValueInChildNode(Node<K, V> node) {
        if (node.getNodes() == null || node.getNodes().isEmpty()) {
            return null;
        }
        List<Entry<K, V>> sortEntries = new ArrayList<>();
        List<Node<K, V>> childNodes = node.getNodes();
        for (Node<K, V> childNode : childNodes) {
            List<Entry<K, V>> entries = childNode.getEntry();
            Entry<K, V> minEntry = entries.get(0);
            sortEntries.add(minEntry);
        }
        Collections.sort(sortEntries);
        return sortEntries.get(0);
    }

    private boolean isNeedMerge(Node<K, V> node) {
        if (node == null) {
            return false;
        }
        List<Entry<K, V>> entries = node.getEntry();
        return entries.size() < rank / 2;
    }

    //判断一个节点是否有富余的键值对
    private boolean isMoreElement(Node<K, V> node) {
        return node != null && (node.getEntry().size() > rank / 2);
    }
}

