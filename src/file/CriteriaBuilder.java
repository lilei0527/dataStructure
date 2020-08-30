package file;

import java.util.Arrays;

/**
 * @author lilei
 **/
@SuppressWarnings("unused")
public class CriteriaBuilder {

    public Criteria eq(FileAttribute attribute, Object value) {
        return new EqualCriteria(attribute, value);
    }

    public Criteria neq(FileAttribute attribute, Object value) {
        return new NotEqualCriteria(attribute, value);
    }

    public Criteria lt(FileAttribute attribute, Object value) {
        return new LessThanCriteria(attribute, value);
    }

    public Criteria ltOrEq(FileAttribute attribute, Object value) {
        return new LessThanOrEqualCriteria(attribute, value);
    }

    public Criteria gt(FileAttribute attribute, Object value) {
        return new GreaterThanCriteria(attribute, value);
    }

    public Criteria gtOrEq(FileAttribute attribute, Object value) {
        return new GreatThanOrEqualCriteria(attribute, value);
    }

    public Criteria like(FileAttribute attribute, Object value) {
        return new LikeCriteria(attribute, value);
    }

    public Criteria and(Criteria ... criteria){
        return new CompositeCriteria(Arrays.asList(criteria),Criteria.BooleanOperator.AND);
    }

    public Criteria or(Criteria ... criteria){
        return new CompositeCriteria(Arrays.asList(criteria),Criteria.BooleanOperator.OR);
    }

    public Criteria parent(Criteria criteria,int parent){
        return new ParentCriteria(criteria,parent);
    }
}
