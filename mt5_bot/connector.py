import logging

import MetaTrader5 as mt5

from . import config

log = logging.getLogger(__name__)


def connect():
    kwargs = {}
    if config.MT5_PATH:
        kwargs["path"] = config.MT5_PATH

    have_creds = config.MT5_LOGIN and config.MT5_PASSWORD and config.MT5_SERVER
    if have_creds:
        # Passing login/password/server straight to initialize() re-authorizes
        # the terminal outright. A bare initialize() followed by a separate
        # login() call can fail with "Authorization failed" if the already-
        # running terminal's existing session is stale/expired -- initialize()
        # itself refuses to attach before login() ever gets a chance to run.
        kwargs.update(login=config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER)

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise RuntimeError(f"No account info after initialize(): {mt5.last_error()}")

    if config.MT5_LOGIN and account.login != config.MT5_LOGIN:
        mt5.shutdown()
        raise RuntimeError(
            f"Connected account (login={account.login}) does not match configured "
            f"MT5_LOGIN={config.MT5_LOGIN} -- refusing to proceed on the wrong account."
        )

    if account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown()
        raise RuntimeError(
            "Safety check failed: connected MT5 account is NOT a demo account "
            f"(login={account.login}, server={account.server}). Refusing to trade."
        )

    if not account.trade_allowed:
        log.warning("Algo trading looks disabled for this account/terminal. "
                    "Enable the 'Algo Trading' button in MT5.")

    log.info("Connected: login=%s server=%s balance=%.2f %s",
              account.login, account.server, account.balance, account.currency)
    return account


def disconnect():
    mt5.shutdown()
