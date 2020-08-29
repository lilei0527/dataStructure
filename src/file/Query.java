package file;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author lilei
 **/
public class Query {
    private List<List<Criteria>> criteriaList = new ArrayList<>();
    private List<BooleanOperator> operator = new ArrayList<>();


    public List<BooleanOperator> getOperator() {
        return operator;
    }

    public enum BooleanOperator {
        AND, OR
    }

    public void andOperator(Criteria... criteria) {
        criteriaList.add(Arrays.asList(criteria));
        for (int i = 0; i < criteria.length - 1; i++) {
            operator.add(BooleanOperator.AND);
        }
    }

    public void orOperator(Criteria... criteria) {
        criteriaList.add(Arrays.asList(criteria));
        for (int i = 0; i < criteria.length - 1; i++) {
            operator.add(BooleanOperator.OR);
        }
    }
}
