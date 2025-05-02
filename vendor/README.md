# Vendored packages required by PyEasyHLA

This is the directory where the Docker build process will look for some required
third-party software.

At the time of writing, we're keeping internally-archived versions of this
software at `[macdatafile]/PyEasyHLA/Vendored software`.

## Oracle Client Libraries

As of the time of writing of this file, we are using Instant Client 23.7 in our
Docker container.  This is only needed for the clinical HLA script, which
uploads data to the database.

- `instantclient-basic-linux.x64-23.7.0.25.01.zip`

This can be downloaded from [Oracle's download website].  As of the time
of writing, Instant Client is typically available for free and without having
to create any user account with the Oracle website.

[Oracle's download website]: https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html

In the future, a later (or even earlier) version of the client might also
work; the specific files you want to use can be specified in the environment
when building the image.  Doing this will likely also require
you to set `PYEASYHLA_ORACLE_HOME` to the appropriate path as well.  For example,
if you're on an ARM64-based Mac and need to use Instant Client 23.3 instead,
`PYEASYHLA_ORACLE_HOME` should be set to `/opt/oracle/instantclient_23_3`
instead of the default.

The Python module we use to connect to an Oracle database, [`python-oracledb`],
has a "thin mode" and a "thick mode".  "Thin mode" has a reduced feature set,
but is able to connect to the database without requiring Instant Client; "thick
mode" uses Instant Client and provides more functionality.  Our code doesn't use
much of this extra functionality, but our production database requires it.

[`python-oracledb`]: https://oracle.github.io/python-oracledb/
