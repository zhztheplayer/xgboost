package ml.dmlc.xgboost4j.java.util;

import java.lang.reflect.Field;

public class UtilReflection {

  private UtilReflection() {

  }

  public static Object getField(Object object, String fieldName) {
    Field field = null;
    try {
      field = object.getClass().getDeclaredField(fieldName);
      field.setAccessible(true);
      return field.get(object);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      throw new RuntimeException(e);
    } finally {
      if (field != null) {
        field.setAccessible(false);
      }
    }
  }
}
