-- Step 8: SQLite schema. Three tables, one-to-many chain:
--     websites 1 --- * comments 1 --- * comment_aspects
--
-- Design notes (viva):
--  * comment_aspects is its own table (not a JSON blob) so we can query
--    "all Negative delivery mentions for site X" and aggregate in SQL/pandas.
--  * Spam comments are STORED (spam_weight = 0) rather than deleted: we keep an
--    audit trail, can show the user why it was held back, and can re-score later.
--  * CHECK constraints make the DB reject impossible values even if Python is buggy.
--  * ON DELETE CASCADE: deleting a website removes its comments and aspects.
--  * pipeline_version records which model produced the scores, because VADER and
--    transformer scores are not directly comparable.

CREATE TABLE IF NOT EXISTS websites (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE
                        CHECK (length(trim(name)) BETWEEN 1 AND 100),
    url         TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS comments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    website_id       INTEGER NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    text             TEXT    NOT NULL CHECK (length(text) BETWEEN 3 AND 2000),
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    spam_prob        REAL    NOT NULL DEFAULT 0   CHECK (spam_prob   BETWEEN 0 AND 1),
    spam_label       TEXT    NOT NULL DEFAULT 'clean'
                             CHECK (spam_label IN ('clean', 'suspicious', 'spam')),
    spam_weight      REAL    NOT NULL DEFAULT 1   CHECK (spam_weight BETWEEN 0 AND 1),
    spam_reasons     TEXT    NOT NULL DEFAULT '[]',          -- JSON list of strings
    polarity         REAL    CHECK (polarity BETWEEN -1 AND 1),  -- mean aspect score; NULL = no aspects
    pipeline_version TEXT
);

CREATE TABLE IF NOT EXISTS comment_aspects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id  INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    aspect      TEXT    NOT NULL CHECK (aspect IN
                    ('genuineness', 'delivery', 'quality', 'support', 'refund', 'pricing')),
    score       REAL    NOT NULL CHECK (score BETWEEN -1 AND 1),
    label       TEXT    NOT NULL CHECK (label IN ('Positive', 'Neutral', 'Negative')),
    evidence    TEXT    NOT NULL DEFAULT '[]',                -- JSON list: the clauses used
    UNIQUE (comment_id, aspect)
);

CREATE INDEX IF NOT EXISTS idx_comments_website ON comments(website_id, created_at);
CREATE INDEX IF NOT EXISTS idx_aspects_comment  ON comment_aspects(comment_id);
CREATE INDEX IF NOT EXISTS idx_aspects_aspect   ON comment_aspects(aspect);
