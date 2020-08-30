package file;

import java.io.File;
import java.util.List;


public class ParentCriteria extends AbstractCriteria {
    private Criteria criteria;
    private int parent; //1表示父路径，2表示父父路径......

    public ParentCriteria(Criteria criteria, int parent) {
        this.criteria = criteria;
        this.parent = parent;
    }

    @Override
    public BooleanOperator getOperator() {
        return Criteria.BooleanOperator.NONE;
    }

    @Override
    public List<Criteria> getExpressions() {
        return null;
    }

    @Override
    public boolean isCompound() {
        return false;
    }

    @Override
    public boolean isMatch(File file) {
        File temp = new File(file.getAbsolutePath());
        for (int i = 0; i < parent; i++) {
            temp = temp.getParentFile();
            if (temp == null) {
                throw new IllegalArgumentException("Parent is too far");
            }
        }
        return criteria.isMatch(temp);
    }
}
