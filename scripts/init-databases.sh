#!/bin/bash
set -e

# Create TalkWithData database and user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create user if not exists
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$TALKWDATA_USER') THEN
            CREATE USER $TALKWDATA_USER WITH PASSWORD '$TALKWDATA_PASSWORD';
        END IF;
    END
    \$\$;

    -- Create database if not exists
    SELECT 'CREATE DATABASE $TALKWDATA_DB OWNER $TALKWDATA_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$TALKWDATA_DB')\gexec

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE $TALKWDATA_DB TO $TALKWDATA_USER;
EOSQL

echo "TalkWithData database '$TALKWDATA_DB' and user '$TALKWDATA_USER' have been created successfully."
