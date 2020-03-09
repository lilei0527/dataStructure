package tree.rbTree;

import tree.Entry;
import tree.TreeNode;
import tree.TwoForkTree;


/**
 * @author lilei
 * create at 2020/2/29 15:35
 * <p>
 * 红黑树定义
 * 1.节点有颜色的属性,根节点必须为黑色
 * 2.所有叶子节点是null节点，且颜色为黑色
 * 3.如果一个节点是红色的那么它的两个节点都是黑色
 * 4.从任何一个节点到其叶子节点的简单路径都包含相同数目的黑色节点
 */
@SuppressWarnings("unused")
public class RbTree<K extends Comparable<K>, V> extends TwoForkTree<K, V> {
    private static final boolean RED = false;
    private static final boolean BLACK = true;

    @Override
    public String toString() {
        return "RbTree{" +
                "root=" + root +
                '}';
    }

    @Override
    public void insert(K k, V v) {
        Entry<K, V> entry = new Entry<>(k, v);
        if (root == null) {
            root = new Node<>(null, entry);
        } else {
            Node<K, V> p = (Node<K, V>) root;
            Node<K, V> parent;
            //迭代
            do {
                parent = p;

                if (isRight(entry, p)) {
                    p = (Node<K, V>) p.right;
                } else if (isLeft(entry, p)) {
                    p = (Node<K, V>) p.left;
                } else {
                    p.entry = entry;
                }
            } while (p != null);


            Node<K, V> node = new Node<>(parent, entry);

            if (isRight(entry, parent)) {
                parent.right = node;
            } else {
                parent.left = node;
            }

            fixAfterInsertion(node);
        }
    }



    @Override
    public V delete(K key) {
        Node<K, V> node = (Node<K, V>) search(root, key);
        if (node == null) {
            return null;
        }
        deleteNode(node);
        return node.entry.value;
    }

    public void deleteNode(Node<K, V> p) {
        //待删除节点有两个孩子
        if (p.left != null && p.right != null) {
            Node<K, V> s =  successor(p);
            //将待删除节点属性赋值到后继节点
            p.entry = s.entry;
            p = s;
        }

        Node<K, V> replacement = (Node<K, V>) (p.left != null ? p.left : p.right);

        if (replacement != null) {
            if (p.parent == null) {
                root = replacement;
            } else if (p == p.parent.left) {
                p.parent.left = replacement;
            } else {
                p.parent.right = replacement;
            }
            replacement.parent = p.parent;

            if (p.color == BLACK) {
                fixAfterDelete(replacement);
            }
        } else if (p.parent == null) {
            root = null;
        } else {
            if (p.color == BLACK)
                fixAfterDelete(p);

            if (p.parent != null) {
                if (p == p.parent.left)
                    p.parent.left = null;
                else if (p == p.parent.right)
                    p.parent.right = null;
                p.parent = null;
            }
        }


    }


    /**
     * @author lilei
     *
     * 四种异常情况
     *
     * 第一种：
     *      b
     *     /\
     *    r  r
     *   /
     *  node
     *
     *
     *  第二种：
     *      b                b
     *     /                /
     *    r       或者     r
     *   /                 \
     *  node               node
     *
     *
     *  第三种：
     *      b
     *     /\
     *    r  r
     *       \
     *       node
     *
     *
     *  第四种：
     *      b                b
     *      \                \
     *       r       或者     r
     *        \              /
     *        node          node
     *
     */
    private void fixAfterInsertion(Node<K, V> node) {
        node.color = RED;

        while (node != root && node.parent.color == RED) {

            if (parentOf(node) == leftOf(parentOf(parentOf(node)))) {
                Node<K, V> y = rightOf(parentOf(parentOf(node)));
                if (colorOf(y) == RED) {
                    //对应第一种情况
                    setColor(parentOf(parentOf(node)), RED);
                    setColor(y, BLACK);
                    setColor(parentOf(node), BLACK);
                    node = parentOf(parentOf(node));
                } else {
                    //对应第二种情况
                    if (rightOf(parentOf(node)) == node) {
                        node = parentOf(node);
                        rotateLeft(node);
                    }
                    setColor(parentOf(node), BLACK);
                    setColor(parentOf(parentOf(node)), RED);
                    rotateRight(parentOf(parentOf(node)));
                }
            } else {
                Node<K, V> y = leftOf(parentOf(parentOf(node)));
                if (colorOf(y) == RED) {
                    //对应第三种情况
                    setColor(parentOf(parentOf(node)), RED);
                    setColor(y, BLACK);
                    setColor(parentOf(node), BLACK);
                    node = parentOf(parentOf(node));
                } else {
                    //对应第四种情况
                    if (leftOf(parentOf(node)) == node) {
                        node = parentOf(node);
                        rotateRight(node);
                    }
                    setColor(parentOf(node), BLACK);
                    setColor(parentOf(parentOf(node)), RED);
                    rotateLeft(parentOf(parentOf(node)));
                }
            }
        }

        ((Node<K, V>) root).color = BLACK;
    }

    /**
     * @author lilei
     *
     */
    private void fixAfterDelete(Node<K, V> x) {
        while (x != root && colorOf(x) == BLACK) {
            if (x == leftOf(parentOf(x))) {
                Node<K, V> sib = rightOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateLeft(parentOf(x));
                    sib = rightOf(parentOf(x));
                }

                if (colorOf(leftOf(sib)) == BLACK &&
                        colorOf(rightOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } else {
                    if (colorOf(rightOf(sib)) == BLACK) {
                        setColor(leftOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateRight(sib);
                        sib = rightOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(rightOf(sib), BLACK);
                    rotateLeft(parentOf(x));
                    x = (Node<K, V>) root;
                }
            } else { // symmetric
                Node<K, V> sib = leftOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateRight(parentOf(x));
                    sib = leftOf(parentOf(x));
                }

                if (colorOf(rightOf(sib)) == BLACK &&
                        colorOf(leftOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } else {
                    if (colorOf(leftOf(sib)) == BLACK) {
                        setColor(rightOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateLeft(sib);
                        sib = leftOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(leftOf(sib), BLACK);
                    rotateRight(parentOf(x));
                    x = (Node<K, V>) root;
                }
            }
        }

        setColor(x, BLACK);
    }

    private void rotateLeft(Node<K, V> p) {
        if (p != null) {
            Node<K, V> r = (Node<K, V>) p.right;
            p.right = r.left;
            if (r.left != null)
                ((Node<K, V>) r.left).parent = p;
            r.parent = p.parent;
            if (p.parent == null)
                root = r;
            else if (p.parent.left == p)
                p.parent.left = r;
            else
                p.parent.right = r;
            r.left = p;
            p.parent = r;
        }
    }

    private void rotateRight(Node<K, V> p) {
        if (p != null) {
            Node<K, V> l = (Node<K, V>) p.left;
            p.left = l.right;
            if (l.right != null) ((Node<K, V>) l.right).parent = p;
            l.parent = p.parent;
            if (p.parent == null)
                root = l;
            else if (p.parent.right == p)
                p.parent.right = l;
            else p.parent.left = l;
            l.right = p;
            p.parent = l;
        }
    }


    private boolean colorOf(Node<K, V> node) {
        return (node == null ? BLACK : node.color);
    }

    private Node<K, V> parentOf(Node<K, V> node) {
        return (node == null ? null : node.parent);
    }

    private void setColor(Node<K, V> node, boolean c) {
        if (node != null)
            node.color = c;
    }

    private Node<K, V> leftOf(Node<K, V> node) {
        return (node == null) ? null : (Node<K, V>) node.left;
    }

    private Node<K, V> rightOf(Node<K, V> node) {
        return (node == null) ? null : (Node<K, V>) node.right;
    }

    protected Node<K, V> successor(Node<K, V> t) {
        if (t == null)
            return null;
        else if (t.right != null) {
            Node<K, V> p = (Node<K, V>) t.right;
            while (p.left != null)
                p = (Node<K, V>) p.left;
            return p;
        } else {
            Node<K, V> p = t.parent;
            Node<K, V> ch = t;
            while (p != null && ch == p.right) {
                ch = p;
                p = p.parent;
            }
            return p;
        }
    }

    /**
     * Returns the predecessor of the specified Entry, or null if no such.
     */
    protected Node<K, V> predecessor(Node<K, V> t) {
        if (t == null)
            return null;
        else if (t.left != null) {
            TreeNode<K, V> p = t.left;
            while (p.right != null)
                p = p.right;
            return (Node<K, V>) p;
        } else {
            Node<K, V> p = t.parent;
            Node<K, V> ch = t;
            while (p != null && ch == p.left) {
                ch = p;
                p = p.parent;
            }
            return p;
        }
    }


}
