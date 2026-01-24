This is a project-specific variant of
[QuadMatrix](https://github.com/m-vokhm/QuadMatrix).

It contains additional client-delegate integration.

The `ClientDelegate` interface, located in the
`com.mvohm.quadmatrix.api` package, is a project-facing callback
interface intended for passing messages from the library to a client
application.

If this library is ever decoupled from the project which still requires this
functionality, the interface can be easily extracted
into a standalone API module or included into the project.
