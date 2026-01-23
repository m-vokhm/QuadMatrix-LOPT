package com.mvohm.quadmatrix.api;

import java.util.List;

/**
 * An interface describing objects used to transmit messages to
 * the client object -- an object that uses a 'server' objects, containing
 * instances of objects implementing this interface. A primitive analog of a listener.
 */
public interface ClientDelegate {

  void addInfoMessage(String... infoText);
  void addInfoMessage(List<String> infoText);
  void addWarning(String... warningText);
  void addWarning(List<String> warningText);
  void addErrorMessage(String... errorText);
  void addErrorMessage(List<String> errorText);

  void showPercent(double percent);

  void sayStarting(String opName);
  void sayDone();

  /** An implementation that does nothing, used to initialize corresponding fields
   * in objects that do not send messages to the client
   */
  ClientDelegate NO_OP = new ClientDelegate() {

    @Override public void addInfoMessage(List<String> infoText) {}
    @Override public void addInfoMessage(String... infoText) {}
    @Override public void addWarning(List<String> warningText) {}
    @Override public void addWarning(String... warningText) {}
    @Override public void addErrorMessage(List<String> errorText) {}
    @Override public void addErrorMessage(String... errorText) {}

    @Override public void showPercent(double percent) {}

    @Override public void sayStarting(String opName) {}
    @Override public void sayDone() {}
  };
}
