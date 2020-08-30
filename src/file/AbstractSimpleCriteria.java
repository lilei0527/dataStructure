package file;

import java.io.File;
import java.util.Collections;
import java.util.List;

public abstract class AbstractSimpleCriteria extends AbstractCriteria implements Visitor{
    protected FileAttribute attribute;
    private static final List<Criteria> NO_EXPRESSIONS = Collections.emptyList();

    public AbstractSimpleCriteria(FileAttribute attribute) {
        this.attribute = attribute;
    }

    @Override
    public BooleanOperator getOperator() {
        return BooleanOperator.NONE;
    }

    @Override
    public final List<Criteria> getExpressions() {
        return NO_EXPRESSIONS;
    }

    @Override
    public boolean isCompound() {
        return false;
    }

    @Override
    public boolean isMatch(File file) {
        switch (attribute) {
            case NAME:
                return visitName(file);
            case SIZE:
                return visitLength(file);
            case TYPE:
                return visitType(file);
            case UPDATE_TIME:
                return visitUpdateTime(file);
            default:
                return false;
        }
    }
}
