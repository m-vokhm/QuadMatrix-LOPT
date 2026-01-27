package com.mvohm.quadmatrix.api;

import java.util.List;

/** An implementation that does nothing, used to initialize corresponding fields
 * in objects that do not send messages to the client
 */

public class NoOpDelegate implements ClientDelegate {

  @Override public void addInfoMessage(List<String> infoText) {}
  @Override public void addInfoMessage(String... infoText) {}
  @Override public void addWarning(List<String> warningText) {}
  @Override public void addWarning(String... warningText) {}
  @Override public void addErrorMessage(List<String> errorText) {}
  @Override public void addErrorMessage(String... errorText) {}

  @Override public void showPercent(double percent) {}

  @Override public void sayStarting(String opName) {}
  @Override public void sayDone() {}
}
