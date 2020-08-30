package file;
import java.io.File;
import java.util.List;

public class CompositeCriteria extends AbstractCriteria {
    private List<Criteria> criteria;
    private BooleanOperator operator;

    public CompositeCriteria(List<Criteria> criteria, BooleanOperator operator) {
        this.criteria = criteria;
        this.operator = operator;
    }



    @Override
    public BooleanOperator getOperator() {
        return operator;
    }

    @Override
    public List<Criteria> getExpressions() {
        return criteria;
    }

    @Override
    public boolean isCompound() {
        return true;
    }

    @Override
    public boolean isMatch(File file) {
        if (operator == BooleanOperator.AND) {
            for (Criteria criteria : criteria) {
                if (!criteria.isMatch(file)) {
                    return false;
                }
            }
            return true;
        } else {
            for (Criteria criteria : criteria) {
                if (criteria.isMatch(file)) {
                    return true;
                }
            }
            return false;
        }
    }
}
